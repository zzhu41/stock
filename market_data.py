# -*- coding: utf-8 -*-
"""行情数据模块：腾讯财经公开接口，纯标准库，本地 CSV 缓存 + 增量更新。

日K: web.ifzq.gtimg.cn/appstock/app/fqkline/get  前复权日线, 单页最多640条, 分页拉全历史
报价: qt.gtimg.cn/q=...                          盘中实时价 / 盘后收盘价
"""
import csv
import json
import math
import os
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 标的池: code -> (名称, 腾讯前缀, 角色)
# 角色: stock=A股竞赛池(受牛熊开关约束), global=跨境池(纯动量, 不受开关约束),
#       gold=黄金备胎, cash=货币停靠
UNIVERSE = {
    "159915": ("创业板ETF",    "sz", "stock"),
    "588080": ("科创50ETF",    "sh", "stock"),
    "510300": ("沪深300ETF",   "sh", "stock"),
    "510500": ("中证500ETF",   "sh", "stock"),
    "563300": ("中证2000ETF",  "sh", "stock"),
    "512400": ("有色金属ETF",  "sh", "stock"),
    "512890": ("红利低波ETF",  "sh", "stock"),
    "513100": ("纳指ETF",      "sh", "global"),
    "513120": ("港股创新药ETF","sh", "global"),
    "518880": ("黄金ETF",      "sh", "gold"),
    "511880": ("货币ETF",      "sh", "cash"),
}
STOCK_POOL = [c for c, v in UNIVERSE.items() if v[2] == "stock"]
GLOBAL_POOL = [c for c, v in UNIVERSE.items() if v[2] == "global"]
GOLD = "518880"
CASH = "511880"
FULL_START = "2010-01-01"  # 全量拉取的起点
QFQ_OVERLAP_ROWS = 21     # 每次更新核对已完成日K, 分红重定基当天即修复

_UA = {"User-Agent": "Mozilla/5.0"}


def _get(url, timeout=20, retries=4, binary=False):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            return raw if binary else raw.decode("utf-8")
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def _kline_page(symbol, start, end):
    """拉 [start, end] 区间内最近 640 根日K，返回 [[date, open, close, volume], ...] 升序。"""
    url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=%s,day,%s,%s,640,qfq"
           % (symbol, start, end))
    d = json.loads(_get(url))
    node = (d.get("data") or {}).get(symbol) or {}
    rows = node.get("qfqday") or node.get("day") or []
    out = []
    for r in rows:
        if len(r) <= 5 or r[5] in ("", None):
            raise ValueError("%s 日K缺成交量" % symbol)
        vol = float(r[5])
        out.append([r[0], float(r[1]), float(r[2]), vol])
    return out


def _date(value):
    if not isinstance(value, str) or datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") != value:
        raise ValueError("非法交易日期: %r" % value)
    return value


def _number(value, label, positive=False):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s 非法: %r" % (label, value))
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError("%s 非法: %r" % (label, value))
    return value


def _validated_rows(rows, end):
    """校验后返回独立的标准化行; 不修补缺失价格/量, 不静默排序或去重。"""
    out = []
    for r in rows:
        if len(r) != 4:
            raise ValueError("日K须有日期/开盘/收盘/成交量四列")
        d = _date(r[0])
        if d > end or (out and d <= out[-1][0]):
            raise ValueError("日K日期越界或未严格递增: %s" % d)
        out.append((d, _number(r[1], "开盘", True), _number(r[2], "收盘", True),
                    _number(r[3], "成交量")))
    return out


def _full_history(symbol, today):
    rows, end = [], today
    while True:
        page = _validated_rows(_kline_page(symbol, FULL_START, end), end)
        if not page:
            if not rows:
                raise ValueError("%s 全量行情为空, 保留原缓存" % symbol)
            break
        rows = page + rows
        if len(page) < 640 or page[0][0] <= FULL_START:
            break
        end = (datetime.strptime(page[0][0], "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        time.sleep(1.2)
    return rows


def _write_history(cache, rows):
    """失败时保留完整旧缓存; rename 前不存在对正式文件的写入。"""
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8",
                                         dir=os.path.dirname(cache), prefix=".history-",
                                         delete=False) as f:
            tmp = f.name
            csv.writer(f).writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, cache)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def fetch_history(code, full=False):
    """前复权日K。增量先核对重叠区间, 复权基准变化则全量重拉后原子替换。
    空响应/非法数据/重拉失败均抛错并保留旧缓存, 不生成基于混合复权价格的信号。"""
    symbol = UNIVERSE[code][1] + code
    cache = os.path.join(DATA_DIR, "%s.csv" % code)
    cached = []
    today = _today()
    if os.path.exists(cache):
        with open(cache, newline="", encoding="utf-8") as f:
            cached = [r for r in csv.reader(f) if r]
    old_last = cached[-1][0] if cached else None
    old_dates = {_date(r[0]) for r in cached}
    rebased = False
    if cached and len(cached[0]) == 3:
        full = True  # 旧格式没有量, 整体更新; 更新失败时旧文件仍保留
    elif cached:
        cached = _validated_rows(cached, today)

    if cached and not full and len(cached) > 1:
        start = cached[-QFQ_OVERLAP_ROWS:][0][0]
        new = _validated_rows(_kline_page(symbol, start, today), today)
        new = [r for r in new if r[0] >= start]
        if not new or new[-1][0] < old_last:
            raise ValueError("%s 增量行情为空或回退, 保留原缓存" % code)
        old_prices = {r[0]: r for r in cached[:-1]}  # 最后缓存行可能仍是盘中K, 不据此判复权漂移
        overlap = [(old_prices[r[0]], r) for r in new if r[0] in old_prices]
        if not overlap:
            raise ValueError("%s 增量缺重叠区间, 无法确认复权基准" % code)
        drifted = any(abs(a[k] / b[k] - 1.0) > 1e-6
                      for a, b in overlap for k in (1, 2))
        if drifted:
            rows = _full_history(symbol, today)
            rebased = True
        else:
            # 保留未返回的旧日期, 避免接口局部缺行抹掉缓存。
            merged = {r[0]: r for r in cached}
            merged.update({r[0]: r for r in new})
            rows = [merged[d] for d in sorted(merged)]
    else:
        rows = _full_history(symbol, today)
        rebased = True
    if old_last and rows[-1][0] < old_last:
        raise ValueError("%s 全量行情日期回退, 保留原缓存" % code)
    if rebased and old_dates - {r[0] for r in rows}:
        raise ValueError("%s 全量行情缺已有历史日期, 保留原缓存" % code)
    _write_history(cache, rows)
    return rows


def fetch_realtime(codes=None, detailed=False):
    """默认返回 {code:(name,price)}; detailed=True 含报价交易日期/时间/开盘/累计成交量。
    时间戳为腾讯的中国市场当地时间; 成交量使用第6字段(手), 与日K量同单位。
    detailed 模式拒绝缺失/非法字段; 交易日是否匹配由 prepare_live_histories 校验。"""
    codes = codes or [c for c in UNIVERSE if c != CASH]
    q = ",".join(UNIVERSE[c][1] + c for c in codes)
    raw = _get("https://qt.gtimg.cn/q=%s" % q, timeout=10, binary=True).decode("gbk")
    out = {}
    for line in raw.strip().split(";"):
        if "=" not in line:
            continue
        f = line.split("=", 1)[1].strip().strip('"').split("~")
        if len(f) <= 3 or f[2] not in codes:
            continue
        code = f[2]
        price = _number(f[3], "%s 实时价" % code, True)
        if not detailed:
            out[code] = (f[1], price)
            continue
        if len(f) <= 30:
            raise ValueError("%s 实时报价缺时间戳" % code)
        stamp = f[30]
        if len(stamp) != 14 or not stamp.isdigit():
            raise ValueError("%s 实时报价时间戳非法" % code)
        dt = datetime.strptime(stamp, "%Y%m%d%H%M%S")
        out[code] = {"name": f[1], "price": price, "date": dt.strftime("%Y-%m-%d"),
                     "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                     "prev_close": _number(f[4], "%s 实时除权前收" % code, True),
                     "open": _number(f[5], "%s 实时开盘" % code, True),
                     "volume": _number(f[6], "%s 实时成交量" % code)}
    if detailed and set(codes) - set(out):
        raise ValueError("缺实时报价: %s" % ",".join(sorted(set(codes) - set(out))))
    return out


def prepare_live_histories(histories, quotes, signal_date, now=None, max_age_seconds=180):
    """纯函数: 报价与信号日期严格匹配, 保留昨日K线并替换/追加当日K线。
    非货币ETF必须有当日报价; CASH 无报价时保持其历史, 有报价时同样严格校验。
    不修改输入, 不允许未来K线, 不用昨日量冒充今日累计量。"""
    signal_date = _date(signal_date)
    now = now or datetime.now()
    if now.strftime("%Y-%m-%d") != signal_date:
        raise ValueError("实时信号日期与当前时钟不一致")
    stamps = []
    out = {}
    for code, history in histories.items():
        rows = _validated_rows(history, signal_date)
        if not rows:
            raise ValueError("%s 历史行情为空" % code)
        quote = quotes.get(code)
        if quote is None and code == CASH:
            out[code] = rows
            continue
        if not isinstance(quote, dict) or quote.get("date") != signal_date:
            raise ValueError("%s 缺 %s 当日报价或报价已陈旧" % (code, signal_date))
        stamp = quote.get("timestamp", "")
        try:
            moment = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
            stamp_date = moment.strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            raise ValueError("%s 报价时间戳非法" % code)
        if stamp_date != signal_date:
            raise ValueError("%s 报价日期与时间戳不一致" % code)
        if not -5 <= (now - moment).total_seconds() <= max_age_seconds:
            raise ValueError("%s 报价时间已过期或来自未来" % code)
        stamps.append(moment)
        row = (signal_date, _number(quote.get("open"), "%s 开盘" % code, True),
               _number(quote.get("price"), "%s 实时价" % code, True),
               _number(quote.get("volume"), "%s 当日累计量" % code))
        if rows[-1][0] == signal_date:
            rows[-1] = row
        else:
            rows.append(row)
        out[code] = rows
    if stamps and (max(stamps) - min(stamps)).total_seconds() > 60:
        raise ValueError("资产报价时间差超过60秒")
    return out


def _today():
    return datetime.now().strftime("%Y-%m-%d")


if __name__ == "__main__":
    for code in UNIVERSE:
        rows = fetch_history(code)
        print("%s %s: %d 行, %s ~ %s, 最新收盘 %.3f"
              % (code, UNIVERSE[code][0], len(rows), rows[0][0], rows[-1][0], rows[-1][2]))
        time.sleep(1.0)
    print("实时:", fetch_realtime())
