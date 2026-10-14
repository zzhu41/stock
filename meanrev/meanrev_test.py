# -*- coding: utf-8 -*-
"""均值回归策略独立回测 —— 与 V12-R2 冻结口径严格可比, 不触碰任何封案文件。

口径(对齐 v12_r2/PROTOCOL.md 与主推送候选 v12r2_08b522c2e1aa291f054d):
  数据: v10_h_close/corrected_snapshots 修正TR净值(全收益, 含分红)
  区间: 2014-01-02 ~ 2026-09-24 (冻结观察日序列, 244观察日/年)
  执行: T日收盘出信号、T日收盘成交; 首评价日建仓免费; 换仓双边各1bp(close_1bp)
  池子: 原11池 —— 10只风险资产 + 511880货币停靠, 与主策略完全相同

三个变体(参数事先固定, 不做全网格寻优; 网格只作稳健性参考):
  A 纯横截面反转:   全场 MOM5 最低者, 滞回缓冲1pp(新候选须比持仓多跌1pp才换)
  B 趋势内回调:     仅 close>MA60 者参与, 选 MOM5 最低; 无合格或持仓跌破年线→货币; 缓冲1pp
  C 绝对超卖回归:   bias10=close/MA10-1<=-3% 才入场; 持仓 bias10>=0 回归完成离场;
                    更深超卖(低1pp)换仓; 无机会持货币
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/stock")
SNAP = ROOT / "v10_h_close" / "corrected_snapshots"
DATES = json.loads((ROOT / "v12_r2" / "results" / "path_metadata.json").read_text())["dates"]
POOL = ["159915", "588080", "510300", "510500", "563300",
        "512400", "512890", "513100", "513120", "518880"]
CASH = "511880"
FEE1 = 0.0001      # 单边1bp, 与 close_1bp 场景一致
TD = 244
WARM = 61          # MA60 成熟门(需60日历史)

NAME = {"159915": "创业板", "588080": "科创50", "510300": "沪深300", "510500": "中证500",
        "563300": "中证2000", "512400": "有色金属", "512890": "红利低波", "513100": "纳指",
        "513120": "港股创新药", "518880": "黄金", "511880": "货币"}


def load_series(code):
    rows = []
    with open(str(SNAP / ("%s.csv" % code))) as f:
        for r in csv.reader(f):
            if r:
                rows.append((r[0], float(r[2])))  # date, close_TR
    return rows


# 对齐到冻结日历: 未上市/缺价为 NaN, 中途缺价前向填充(罕见, 影响可忽略)
CLOSE = {}
for c in POOL + [CASH]:
    px = dict(load_series(c))
    arr = np.array([px.get(d, np.nan) for d in DATES])
    last = np.nan
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            arr[i] = last
        else:
            last = arr[i]
    CLOSE[c] = arr

RET = {}
for c in POOL + [CASH]:
    r = np.zeros(len(DATES))
    r[1:] = CLOSE[c][1:] / CLOSE[c][:-1] - 1.0
    r[np.isnan(r)] = 0.0
    RET[c] = r


def rolling_mean(x, n):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


MOM = {n: {c: CLOSE[c] / np.roll(CLOSE[c], n) - 1.0 for c in POOL} for n in (3, 5, 10)}
for n in (3, 5, 10):
    for c in POOL:
        MOM[n][c][:n] = np.nan
MA60 = {c: rolling_mean(CLOSE[c], 60) for c in POOL}
MA10 = {c: rolling_mean(CLOSE[c], 10) for c in POOL}
BIAS10 = {c: CLOSE[c] / MA10[c] - 1.0 for c in POOL}


def pick_A(i, holding, mom_n, buf):
    vals = [(c, MOM[mom_n][c][i]) for c in POOL if not np.isnan(MOM[mom_n][c][i])]
    if not vals:
        return CASH
    best = min(vals, key=lambda x: x[1])
    if holding != CASH and not np.isnan(MOM[mom_n][holding][i]):
        if MOM[mom_n][holding][i] <= best[1] + buf:
            return holding
    return best[0]


def pick_B(i, holding, mom_n, buf):
    vals = [(c, MOM[mom_n][c][i]) for c in POOL
            if not np.isnan(MOM[mom_n][c][i]) and not np.isnan(MA60[c][i])
            and CLOSE[c][i] > MA60[c][i]]
    if not vals:
        return CASH
    best = min(vals, key=lambda x: x[1])
    if (holding != CASH and not np.isnan(MOM[mom_n][holding][i])
            and not np.isnan(MA60[holding][i]) and CLOSE[holding][i] > MA60[holding][i]):
        if MOM[mom_n][holding][i] <= best[1] + buf:
            return holding
    return best[0]


def pick_C(i, holding, thr, buf):
    vals = [(c, BIAS10[c][i]) for c in POOL
            if not np.isnan(BIAS10[c][i]) and BIAS10[c][i] <= thr]
    best = min(vals, key=lambda x: x[1]) if vals else None
    if holding != CASH:
        b = BIAS10[holding][i]
        if np.isnan(b) or b >= 0.0:            # 回归完成, 离场
            return best[0] if best else CASH
        if best and best[1] < b - buf:          # 更深超卖者, 换
            return best[0]
        return holding
    return best[0] if best else CASH


def run(pick):
    holding = CASH
    navs = np.ones(len(DATES))
    switches = 0
    hold_days = []
    cur_days = 0
    for i in range(1, len(DATES)):
        nav = navs[i - 1] * (1.0 + RET[holding][i])
        cur_days += 1
        if i >= WARM:
            target = pick(i, holding)
            if target != holding and not np.isnan(CLOSE[target][i]):
                nav *= 1.0 - 2.0 * FEE1
                switches += 1
                hold_days.append(cur_days)
                cur_days = 0
                holding = target
        navs[i] = nav
    hold_days.append(cur_days)
    rets = np.diff(navs) / navs[:-1]
    return np.insert(rets, 0, 0.0), switches, hold_days, navs


def metrics(rets):
    r = np.asarray(rets, float)
    nav = np.cumprod(1.0 + r)
    n = len(r)
    cagr = nav[-1] ** (TD / n) - 1.0
    peak = np.maximum.accumulate(np.insert(nav, 0, 1.0))[1:]
    mdd = (nav / peak - 1.0).min()
    sd = r.std()
    return dict(nav=nav[-1], cagr=cagr, max_dd=mdd,
                sharpe=(r.mean() / sd * TD ** 0.5) if sd > 0 else 0.0,
                calmar=(cagr / abs(mdd)) if mdd < 0 else 0.0)


def yearly(rets):
    out = {}
    for y in sorted({d[:4] for d in DATES}):
        mask = np.array([d[:4] == y for d in DATES])
        out[y] = np.prod(1.0 + rets[mask]) - 1.0
    return out


def fmt(name, m, sw, hold_days):
    avg_hold = np.mean(hold_days) if hold_days else 0.0
    return ("%-26s 年化 %+6.2f%% | 回撤 %6.2f%% | 夏普 %4.2f | 卡玛 %4.2f | 换手 %4d次(年均%4.1f) | 均持 %4.1f天"
            % (name, m["cagr"] * 100, m["max_dd"] * 100, m["sharpe"], m["calmar"],
               sw, sw / (len(DATES) / TD), avg_hold))


def main():
    # 基线: 主推送候选 close_1bp 日收益
    meta = json.loads((ROOT / "v12_r2" / "results" / "path_metadata.json").read_text())
    idx = meta["ids"].index("v12r2_08b522c2e1aa291f054d")
    with np.load(str(ROOT / "v12_r2" / "results" / "paths.npz"), allow_pickle=False) as z:
        r2 = z["close_1bp__returns"][idx]
    m2 = metrics(r2)
    print("口径: %s ~ %s, %d 观察日, close_1bp(双边万二), 修正TR" % (DATES[0], DATES[-1], len(DATES)))
    print(fmt("基线 V12-R2 主推送(动量)", m2, -1, []))
    print("  逐年:", " ".join("%s:%+.1f%%" % (y, r * 100) for y, r in yearly(r2).items()))

    results = {}
    mains = [
        ("A 纯横截面反转 mom5+buf1", lambda i, h: pick_A(i, h, 5, 0.01)),
        ("B 趋势内回调 MA60+mom5", lambda i, h: pick_B(i, h, 5, 0.01)),
        ("C 绝对超卖 bias10<=-3%", lambda i, h: pick_C(i, h, -0.03, 0.01)),
    ]
    for name, pick in mains:
        rets, sw, hd, _ = run(pick)
        m = metrics(rets)
        results[name] = (rets, m)
        print(fmt(name, m, sw, hd))
        print("  逐年:", " ".join("%s:%+.1f%%" % (y, r * 100) for y, r in yearly(rets).items()))
        corr = np.corrcoef(rets[1:], r2[1:])[0, 1]
        mix = metrics((rets + r2) / 2.0)
        print("  与V12-R2日收益相关 %+.2f | 50/50混合: 年化 %+6.2f%% 回撤 %6.2f%% 夏普 %4.2f 卡玛 %4.2f"
              % (corr, mix["cagr"] * 100, mix["max_dd"] * 100, mix["sharpe"], mix["calmar"]))

    print("\n== 稳健性网格(仅参考, 非寻优) ==")
    print("-- A 纯反转: mom窗口 × 缓冲 --")
    for n in (3, 5, 10):
        for buf in (0.0, 0.01):
            rets, sw, hd, _ = run(lambda i, h: pick_A(i, h, n, buf))
            m = metrics(rets)
            print(fmt("A mom%d buf%.2f" % (n, buf), m, sw, hd))
    print("-- B 趋势过滤: mom窗口 × 缓冲 --")
    for n in (3, 5, 10):
        for buf in (0.0, 0.01):
            rets, sw, hd, _ = run(lambda i, h: pick_B(i, h, n, buf))
            m = metrics(rets)
            print(fmt("B mom%d buf%.2f" % (n, buf), m, sw, hd))
    print("-- C 绝对超卖: 阈值 --")
    for thr in (-0.02, -0.03, -0.05):
        rets, sw, hd, _ = run(lambda i, h: pick_C(i, h, thr, 0.01))
        m = metrics(rets)
        print(fmt("C thr%.2f" % thr, m, sw, hd))


if __name__ == "__main__":
    main()
