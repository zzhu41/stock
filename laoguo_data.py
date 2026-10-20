# -*- coding: utf-8 -*-
"""牢国策略(github.com/marcocrxu/ETFMomentum Top-1 动量轮动)的数据适配层。

数据源(与 /root/stock 现有系统同源):
- ETF 日线: data/{code}.csv 第3列(腾讯前复权收盘, 每日由现有系统更新)
- 指数日线: 腾讯 kline 接口, 本地缓存 data/idx_{symbol}.csv (date,close), 增量更新;
  指数无复权, 当日盘中行不入缓存(与 ETF qfq 当日延迟同处理)
- QQQ-CNY: ext_data 的 usQQQ.csv(新浪) × whUSDCNY.csv(腾讯), cron 14:40 刷新
- 当日实时价: market_data.fetch_realtime (腾讯)

口径与 ETFMomentum 的 live/runner.py 严格一致:
- 价格比动量 m[w:] = (arr[w:]/arr[:-w]-1) × 100
- 纳指混合: mom = (1-0.7)×mom(513100) + 0.7×mom(QQQ×USDCNY)
- QQQ 无前视对齐: CN 交易日 D 只用美东日期 ≤ D-1 的最后收盘
"""
import bisect
import csv
import json
import os
import time
import urllib.request

import market_data  # 复用实时价与校验(纯标准库)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
QQQ_CSV = os.path.join(DATA_DIR, "usQQQ.csv")
FX_CSV = os.path.join(DATA_DIR, "whUSDCNY.csv")
INDEX_CACHE = os.path.join(DATA_DIR, "idx_%s.csv")  # % symbol, 如 sh000001

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# 牢国策略固定池: 名称 -> ETF 代码(与 ETFMomentum web/bt_runner.py 默认池一致)
POOL = {"纳指": "513100", "黄金": "518880", "创业": "159915"}
# 指数: 键 -> 腾讯 symbol
INDICES = {"上证": "sh000001", "创业指": "sz399006"}

QQQ_WEIGHT = 0.7  # 纳指动量混合的 QQQ 权重(仓库 SignalConfig.nz_mom_qqq_w 默认)
HISTORY_START = "2016-07-07"  # 数据起点, 与仓库 live/runner.py 的 start_all 严格一致:
                              # 加速止盈阈值是 expanding 全历史展开, 起点不同则阈值不同


def _read_close_csv(path):
    """读 (date, close) 两列 CSV, 返回升序 list[tuple]。"""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if r:
                rows.append((r[0], float(r[1])))
    return rows


def load_etf_close(code):
    """本地腾讯 qfq 日线 → [(date, close)] 升序; CSV 列 = date,open,close,volume。"""
    path = os.path.join(DATA_DIR, "%s.csv" % code)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if r:
                rows.append((r[0], float(r[2])))
    return rows


def _kline_page(symbol, start, end):
    """腾讯日K一页(≤640根): [[date, open, close, ...], ...] 升序。"""
    url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=%s,day,%s,%s,640,day"
           % (symbol, start, end))
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read().decode("utf-8"))
    node = (d.get("data") or {}).get(symbol) or {}
    return node.get("day") or node.get("qfqday") or []


def refresh_index(symbol, today, _sleep=time.sleep):
    """增量更新指数缓存并返回 [(date, close)] 升序。

    只落盘已完结的日K(最后一行日期 < today 才收录); 网络/数据失败保留旧缓存。
    """
    path = INDEX_CACHE % symbol
    cached = _read_close_csv(path) if os.path.exists(path) else []
    start = "2013-01-01"
    if cached:
        start = cached[-1][0]
    if cached and start >= today:
        return cached  # 缓存已含最近完结日
    pages = []
    end = today
    while True:
        rows = _kline_page(symbol, start, end)
        if not rows:
            break
        pages = rows + pages
        if len(rows) < 640:
            break
        end = time.strftime("%Y-%m-%d",
                            time.localtime(time.mktime(time.strptime(rows[0][0], "%Y-%m-%d")) - 86400))
        _sleep(1.0)
    new = [(r[0], float(r[2])) for r in pages if r[0] < today]  # 当日盘中行不入缓存
    if not new and not cached:
        raise ValueError("%s 指数历史拉取为空" % symbol)
    merged = dict(cached)
    for d, c in new:
        if d >= start:
            merged[d] = c
    rows = sorted(merged.items())
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    os.replace(tmp, path)
    return rows


def load_index_close(symbol, today, offline=False):
    """指数 [(date, close)]; offline=True 时只读缓存(回测/测试用)。"""
    if offline:
        return _read_close_csv(INDEX_CACHE % symbol)
    return refresh_index(symbol, today)


def load_qqq_cny():
    """usQQQ × whUSDCNY → [(us_date, qqq_cny)] 升序; 每根 QQQ 配 ≤ 当日最新汇率。"""
    qqq = _read_close_csv(QQQ_CSV)
    fx = _read_close_csv(FX_CSV)
    if not qqq or not fx:
        raise ValueError("QQQ 或汇率缓存为空(需 cron 14:40 刷新)")
    fx_dates = [d for d, _ in fx]
    out = []
    for d, c in qqq:
        i = bisect.bisect_right(fx_dates, d) - 1
        if i >= 0:
            out.append((d, c * fx[i][1]))
    return out


def align_qqq_to_cn(cn_dates, qqq_rows):
    """CN 交易日 → QQQ-CNY: 每日取美东日期 ≤ D-1 的最后收盘(与仓库 usfeed.align_to_cn 同语义)。"""
    us_dates = [d for d, _ in qqq_rows]
    out = []
    prev = None
    for d in cn_dates:
        # 美东日期严格 < D: bisect_right(d) - 1 即 ≤ D-1 的最后位置(日期粒度按天)
        i = bisect.bisect_right(us_dates, d) - 1
        while i >= 0 and us_dates[i] >= d:
            i -= 1
        out.append(qqq_rows[i][1] if i >= 0 else (prev if prev is not None else float("nan")))
        if i >= 0:
            prev = qqq_rows[i][1]
    return out


def fetch_index_realtime():
    """腾讯实时指数价 → {"上证": price, "创业指": price}; 与仓库 live/runner.py 同接口同字段。"""
    url = "http://qt.gtimg.cn/q=s_sh000001,s_sz399006"
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=5) as r:
        text = r.read().decode("gbk", "ignore")
    out = {}
    for line in text.strip().split("\n"):
        parts = line.split("~")
        if len(parts) < 4:
            continue
        name = {"s_sh000001": "上证", "s_sz399006": "创业指"}.get(parts[0].split("=")[0].replace("v_", "").strip())
        if name:
            try:
                out[name] = float(parts[3])
            except (TypeError, ValueError):
                pass
    return out


def _momentum(arr, w):
    """价格比动量(仓库 live/runner.py 口径): m[t] = (arr[t]/arr[t-w]-1)×100。"""
    n = len(arr)
    m = [float("nan")] * n
    for t in range(w, n):
        if arr[t - w] == arr[t - w] and arr[t] == arr[t] and arr[t - w] > 0:  # NaN 检查
            m[t] = (arr[t] / arr[t - w] - 1.0) * 100.0
    return m


def build_inputs(today, offline=False, live_quote=None):
    """组装 compute_signals 的全部输入(纯 list, 调用方转 numpy)。

    today: 当日日期 YYYY-MM-DD。offline=True 不触网络(仅用缓存, 回测/测试)。
    live_quote: 注入实时价 {名称: price} 钩子(测试); None 时按 offline 决定是否拉腾讯实时。
    返回 dict(etf_names, dates, price_mat, mom20_mat, mom10_mat, sh_mom, cyb_p, sh_p)
    """
    etf_names = list(POOL)
    series = {nm: dict(load_etf_close(code)) for nm, code in POOL.items()}
    sh = dict(load_index_close(INDICES["上证"], today, offline))
    cyb = dict(load_index_close(INDICES["创业指"], today, offline))

    dates = sorted(set().union(*[set(s) for s in series.values()], set(sh)))
    dates = [d for d in dates if HISTORY_START <= d < today]
    if live_quote is None and not offline:
        rt = market_data.fetch_realtime()
        # 腾讯代码 → 名称(与仓库 live/runner.py 的映射一致)
        by_code = {"513100": "纳指", "518880": "黄金", "159915": "创业"}
        live_quote = {}
        for code, quote in rt.items():
            nm = by_code.get(code)
            if nm and quote[1] > 0:
                live_quote[nm] = quote[1]
    if live_quote is not None and not offline:
        # 指数实时价必须补齐: 上证空仓过滤与恐慌抄底的当日判定依赖它们,
        # 缺失时信号层当日静默(永远无动作), 比报错更危险。
        if "上证" not in live_quote or "创业指" not in live_quote:
            idx_rt = fetch_index_realtime()
            live_quote = dict(live_quote)
            live_quote.setdefault("上证", idx_rt.get("上证"))
            live_quote.setdefault("创业指", idx_rt.get("创业指"))
    if live_quote:
        dates = dates + [today]

    def col(table):
        return [table.get(d, float("nan")) for d in dates]

    price_mat = [col(series[nm]) for nm in etf_names]
    sh_p = col(sh)
    cyb_p = col(cyb)
    if live_quote:
        for i, nm in enumerate(etf_names):
            if nm in live_quote:
                price_mat[i][-1] = live_quote[nm]
        if live_quote.get("上证"):
            sh_p[-1] = live_quote["上证"]
        if live_quote.get("创业指"):
            cyb_p[-1] = live_quote["创业指"]

    mom20_mat = [_momentum(row, 20) for row in price_mat]
    mom10_mat = [_momentum(row, 10) for row in price_mat]
    sh_mom = _momentum(sh_p, 20)

    # 纳指动量混合: (1-w)×ETF + w×QQQ-CNY (仓库默认开; QQQ 缺失当日回退纯 ETF 动量)
    qqq = align_qqq_to_cn(dates, load_qqq_cny())
    qm20 = _momentum(qqq, 20)
    qm10 = _momentum(qqq, 10)
    nz = etf_names.index("纳指")
    for t in range(len(dates)):
        if qm20[t] == qm20[t]:  # 非 NaN 才混合
            mom20_mat[nz][t] = (1 - QQQ_WEIGHT) * mom20_mat[nz][t] + QQQ_WEIGHT * qm20[t] \
                if mom20_mat[nz][t] == mom20_mat[nz][t] else float("nan")
        if qm10[t] == qm10[t]:
            mom10_mat[nz][t] = (1 - QQQ_WEIGHT) * mom10_mat[nz][t] + QQQ_WEIGHT * qm10[t] \
                if mom10_mat[nz][t] == mom10_mat[nz][t] else float("nan")

    return dict(etf_names=etf_names, dates=dates, price_mat=price_mat,
                mom20_mat=mom20_mat, mom10_mat=mom10_mat,
                sh_mom=sh_mom, cyb_p=cyb_p, sh_p=sh_p)
