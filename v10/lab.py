# -*- coding: utf-8 -*-
"""v10 研究实验台：不改 v9.1 任何文件，monkey-patch 方式挂接候选机制（项目惯例, 同 v9_robustness.py）。

安全设计:
  - 数据走 data/*.csv 只读快照(无网络/无缓存写, 多进程并发安全);
  - 所有变体由 CFG 开关驱动; use_copies 关闭时完全走 v9.1 原始路径;
  - decide/_enter_ok/_exit_hit 的 v10 副本逐行复刻 strategy.py, 钩子默认惰性;
  - "copy_check" 变体(副本全开、候选全关)必须与 baseline 逐位一致 —— 副本保真校验。

候选方向(坟墓场之外的空白点, 用户指定不空仓簇为重点):
  不空仓簇: ne_score0(兜底须score>0) / ne_gold_first(兜底黄金优先) / ne_buf2(避险池内轮动缓冲x2)
  缓冲簇:   scorebuf(score相对差替代MOM20差) / bearbuf(熊市缓冲加倍) / crosspool(跨池切换加罚)
           / rankconfirm(挑战者连续N天确认才轮动)
  合一簇:   enter_slope/exit_slope(进出场合一于WLS斜率符号)
  其他:     gold_exempt(黄金豁免熊门) / wls2030(双窗集成)
"""
import csv
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import strategy
import backtest
import market_data
from market_data import UNIVERSE, GOLD, CASH

DATA_DIR = os.path.join(BASE, "data")
EXTRA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_extra")
# 第三轮新增标的(ext_fetch.py 拉取); 进程内存内注册进 UNIVERSE(不写盘, 实盘进程不受影响)
EXTRA_UNIVERSE = {
    "510880": ("红利ETF", "sh", "stock"),
    "515080": ("中证红利ETF", "sh", "stock"),
    "515100": ("红利低波100ETF", "sh", "stock"),
    "511010": ("国债ETF", "sh", "stock"),
    "513030": ("德国ETF", "sh", "global"),
    "513520": ("日经ETF", "sh", "global"),
    "159985": ("豆粕ETF", "sz", "global"),
}
for _c, _v in EXTRA_UNIVERSE.items():
    market_data.UNIVERSE.setdefault(_c, _v)
BASE_CODES = ["159915", "588080", "510300", "510500", "563300", "512400", "512890",
              "513100", "513120", "518880", "511880"]
CFG = {}                                    # 变体开关(每 run 重置)
STATE = {"last_date": None, "holding": None, "pending": None, "count": 0}  # rankconfirm 状态
TRACE = []                                  # trace 开关时的逐日状态记录

H = None                                    # 快照行情
CAL = None                                  # 交易日历(510300)


def _cfg(k, d=None):
    return CFG.get(k, d)


def load_histories():
    """只读快照: 直接读 data/*.csv + v10/data_extra/*.csv, 不触网络, 不写缓存。
    新增标的不在任何默认池子里(池列表是 import 时建的), 对基线完全惰性。"""
    out = {}
    for code in BASE_CODES:
        with open(os.path.join(DATA_DIR, "%s.csv" % code), newline="", encoding="utf-8") as f:
            out[code] = [(r[0], float(r[1]), float(r[2]), float(r[3]))
                         for r in csv.reader(f) if r]
    for code in EXTRA_UNIVERSE:
        path = os.path.join(EXTRA_DIR, "%s.csv" % code)
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as f:
                out[code] = [(r[0], float(r[1]), float(r[2]), float(r[3]))
                             for r in csv.reader(f) if r]
    return out


def init_data():
    global H, CAL
    if H is None:
        H = load_histories()
        CAL = [r[0] for r in H["510300"]]


# ---------------------------------------------------------------- WLS 斜率(wls2030/r2full 用)
def _wls_slope_ann(seg):
    """复刻 strategy._wls 的年化斜率部分(时间加权回归)。"""
    n = len(seg)
    xs = list(range(n))
    wts = [i + 1 for i in xs]
    wsum = sum(wts)
    wmx = sum(wts[i] * xs[i] for i in xs) / wsum
    wmy = sum(wts[i] * seg[i] for i in xs) / wsum
    wsxy = sum(wts[i] * (xs[i] - wmx) * (seg[i] - wmy) for i in xs)
    wsxx = sum(wts[i] * (xs[i] - wmx) ** 2 for i in xs)
    return (wsxy / wsxx) / wmy * 250 if wsxx > 0 and wmy > 0 else 0.0


def _wls_full(seg):
    """复刻 strategy._wls: 返回 (年化斜率, R²)。权重同为 1..n 线性递增(斜率与R²口径一致=V2)。"""
    n = len(seg)
    xs = list(range(n))
    wts = [i + 1 for i in xs]
    wsum = sum(wts)
    wmx = sum(wts[i] * xs[i] for i in xs) / wsum
    wmy = sum(wts[i] * seg[i] for i in xs) / wsum
    wsxy = sum(wts[i] * (xs[i] - wmx) * (seg[i] - wmy) for i in xs)
    wsxx = sum(wts[i] * (xs[i] - wmx) ** 2 for i in xs)
    wsyy = sum(wts[i] * (seg[i] - wmy) ** 2 for i in xs)
    r2 = (wsxy ** 2) / (wsxx * wsyy) if wsxx > 0 and wsyy > 0 else 0.0
    sl = (wsxy / wsxx) / wmy if wsxx > 0 and wmy > 0 else 0.0
    return sl * 250, r2


def _cifang_score(closes, n=25):
    """次方量化原式: 对数价格 WLS(权重1→2线性), score = (exp(日斜率*250)-1) * R²。"""
    import math
    seg = [math.log(c) for c in closes[-n:]]
    m = len(seg)
    xs = list(range(m))
    wts = [1.0 + i / (m - 1.0) for i in xs]     # 1→2 线性递增(比我们的 1..25 平缓)
    wsum = sum(wts)
    wmx = sum(wts[i] * xs[i] for i in xs) / wsum
    wmy = sum(wts[i] * seg[i] for i in xs) / wsum
    wsxy = sum(wts[i] * (xs[i] - wmx) * (seg[i] - wmy) for i in xs)
    wsxx = sum(wts[i] * (xs[i] - wmx) ** 2 for i in xs)
    wsyy = sum(wts[i] * (seg[i] - wmy) ** 2 for i in xs)
    r2 = (wsxy ** 2) / (wsxx * wsyy) if wsxx > 0 and wsyy > 0 else 0.0
    sl = wsxy / wsxx if wsxx > 0 else 0.0
    return (math.exp(sl * 250) - 1.0) * r2


# ---------------------------------------------------------------- indicators 包装
_orig_indicators = strategy.indicators


def indicators_v10(closes, volumes=None):
    out = _orig_indicators(closes, volumes)
    if out is not None and _cfg("wls2030"):
        # v10: WLS 20/30 双窗斜率等权集成(23-28 平台两端降噪)
        s = (_wls_slope_ann(closes[-20:]) + _wls_slope_ann(closes[-30:])) / 2.0
        out["score"] = s / out["vol"] if out["vol"] > 0 else 0.0
    if out is not None and _cfg("score_r2_full"):
        # v10(借鉴次方量化): score = 年化斜率 × R²(全强度; 第十二轮只测过半强度 0.5+0.5R² 被否)
        s, r2 = _wls_full(closes[-25:])
        out["score"] = s * r2 / out["vol"] if out["vol"] > 0 else 0.0
    if out is not None and _cfg("score_cifang"):
        # v10(次方量化原式): 对数价 WLS(权重1→2) × R², 不除 vol
        out["score"] = _cifang_score(closes)
    return out


strategy.indicators = indicators_v10

# ---------------------------------------------------------------- rank 包装(跨回测重置状态)
_orig_rank = strategy.rank


def rank_v10(histories, on_date=None, live_prices=None):
    if on_date is not None:
        if STATE["last_date"] is not None and on_date < STATE["last_date"]:
            STATE["pending"], STATE["count"] = None, 0   # 新回测开始, 重置确认状态
        STATE["last_date"] = on_date
    return _orig_rank(histories, on_date=on_date, live_prices=live_prices)


strategy.rank = rank_v10


# ---------------------------------------------------------------- _enter_ok 副本 + 钩子
def _enter_ok_v10(ind, code=None, bull=True):
    """逐行复刻 strategy._enter_ok; 钩子: gold_exempt / bear_gate_exempt / enter_slope。"""
    mom_floor = -0.02 if (strategy.KDJ_DIP_BUY and ind["j_val"] < 10) else strategy.ENTER_MOM_MIN
    if not bull and strategy.BEAR_ENTER_MOM > mom_floor:
        # v10: 熊门豁免集(gold_exempt=黄金; bear_gate_exempt=指定防御标的)
        _exempt = ({GOLD} if _cfg("gold_exempt") else set()) | set(_cfg("bear_gate_exempt", ()))
        if code not in _exempt:
            mom_floor = strategy.BEAR_ENTER_MOM
    if _cfg("enter_slope"):
        # v10: 牛市进场门槛由 MOM20>0 改为 WLS score>0(与排名指标对齐); 熊市保留熊门(动量口径)
        if not bull and ind["mom20"] <= mom_floor:
            return False
        if ind["score"] <= 0:
            return False
    elif ind["mom20"] <= mom_floor:
        return False
    if ind["mom20"] > strategy.OVERHEAT and ind["mom5"] <= 0:
        return False
    if strategy.ENTER_VOL_MAX < 9 and ind["vol"] * 244 ** 0.5 > strategy.ENTER_VOL_MAX:
        return False
    if strategy.DUAL_MOM and ind["mom60"] <= 0:
        return False
    if ind["pos_frac"] < strategy.POS_FRAC_MIN:
        return False
    if ind["max_ret"] > strategy.MAX_CAP:
        return False
    if strategy.MACD_FILTER and ind["dif"] <= ind["dea"]:
        return False
    if strategy.MACD_DIF_POS and ind["dif"] <= 0:
        return False
    if strategy.KDJ_GOLDEN and ind["k_val"] <= ind["d_val"]:
        return False
    if strategy.KDJ_NOCHASE and ind["j_val"] > strategy.KDJ_NC_THR:
        return False
    if strategy.RSI_NOCHASE and ind["rsi"] > strategy.RSI_NC_THR:
        return False
    if strategy.RSI_GT50 and ind["rsi"] <= 50:
        return False
    if strategy.MA_ALIGN and not ind["ma_align"]:
        return False
    if strategy.ABOVE_MA20 and not ind["above_ma20"]:
        return False
    if strategy.MA20_RISING and not ind["ma20_rising"]:
        return False
    if ind["bias20"] > strategy.BIAS_NOCHASE:
        return False
    if ind["pctb"] > strategy.PCTB_NOCHASE:
        return False
    if ind["cci"] > strategy.CCI_NOCHASE:
        return False
    if strategy.VOL_RATIO_MAX > 0 and ind["vol_ratio"] > strategy.VOL_RATIO_MAX:
        return False
    if ind["vol_in_ratio"] < strategy.VOL_IN_MIN:
        return False
    if ind["vol_in_ratio"] > strategy.VOL_IN_MAX:
        return False
    if code in strategy.GLOBAL_POOL:
        if strategy.GLOBAL_MA_FILTER and not ind["above_ma"]:
            return False
        if strategy.GLOBAL_MOM60_FILTER and ind["mom60"] <= 0:
            return False
    return True


# ---------------------------------------------------------------- _exit_hit 副本 + 钩子
def _exit_hit_v10(ind):
    """逐行复刻 strategy._exit_hit; 钩子: exit_slope(score转负替代mom20转负)。"""
    if _cfg("exit_slope"):
        if ind["score"] <= 0:
            return "WLS score 转负"
    elif ind["mom20"] <= strategy.EXIT_MOM_FLOOR:
        return "MOM20 低于阈值 %.1f%%" % (strategy.EXIT_MOM_FLOOR * 100) if strategy.EXIT_MOM_FLOOR > 0 else "MOM20 转负"
    if strategy.PANIC_DROP > 0 and ind["ret1"] <= -strategy._panic_thr(ind):
        return "单日急跌 %.1f%% 紧急离场" % (ind["ret1"] * 100)
    if ind["mom20_max"] > strategy.OVERHEAT and ind["mom5"] <= 0:
        return "过热回落(近%d日MOM20峰值 %+.0f%% 且 MOM5 转负)" % (
            strategy.OVERHEAT_LOOKBACK, ind["mom20_max"] * 100)
    if strategy.MOM_DECAY < 9 and ind["mom20_peak20"] - ind["mom20"] > strategy.MOM_DECAY:
        return "动量衰减(峰值 %+.1f%% -> %+.1f%%)" % (ind["mom20_peak20"] * 100, ind["mom20"] * 100)
    if strategy.DONCHIAN_N > 0 and not ind["above_donchian"]:
        return "跌破近%d日最低收盘(唐奇安)" % strategy.DONCHIAN_N
    if strategy.MACD_EXIT and ind["dif"] < ind["dea"]:
        return "MACD 死叉"
    if strategy.KDJ_DEAD_EXIT and ind["k_val"] < ind["d_val"]:
        return "KDJ 死叉"
    if strategy.RSI_LT50_EXIT and ind["rsi"] < 50:
        return "RSI 跌破 50"
    if strategy.ZSCORE_ACCEL and ind["mom10_z"] > 3.0:
        return "加速止盈(mom10 z=%.1f)" % ind["mom10_z"]
    return None


# ---------------------------------------------------------------- decide 副本 + 钩子
def _decide_v10_impl(table, holding, holding_days=99):
    """逐行复刻 strategy.decide; 钩子: ne_score0 / ne_gold_first / ne_buf2 /
    scorebuf / bearbuf / crosspool / rankconfirm。"""
    info = {c: ind for c, ind in table}
    bull = strategy._is_bull(info)
    # v10 池子钩子(默认全空=原样):
    #   defensive: 防御角色(牛熊竞赛池均可参赛 + 熊市不制度性离场), 如红利低波防御化
    #   bear_extra: 仅熊市竞赛池(不牛市参赛) + 不制度性离场, 如国债防御层
    #   stock_extra: 额外A股池成员(牛市参赛, 熊市制度性离场)
    #   global_extra: 额外跨境池成员(牛熊均参赛 + 兜底池 + 缓冲按3%), 如德国/日经/豆粕
    #   fb_extra: 额外不空仓兜底池成员
    _DEF = set(_cfg("defensive", ()))
    _BEX = set(_cfg("bear_extra", ()))
    _STX = set(_cfg("stock_extra", ()))
    _FBX = set(_cfg("fb_extra", ()))
    _GLX = set(_cfg("global_extra", ()))
    _BASE_COMP = set(strategy.STOCK_POOL) | set(strategy.GLOBAL_POOL)
    if bull or strategy.BEAR_OPEN_STOCK:
        comp_pool = strategy.STOCK_POOL + strategy.GLOBAL_POOL + sorted((_DEF | _STX | _GLX) - _BASE_COMP)
    else:
        comp_pool = strategy.GLOBAL_POOL + [GOLD] + sorted(
            (_DEF | _BEX | _GLX) - set(strategy.GLOBAL_POOL) - {GOLD})
    best = next((t for t in table if t[0] in comp_pool and _enter_ok_v10(t[1], t[0], bull)), None)

    if best:
        target, reason = best[0], "竞赛池动量第一" if bull else "熊市体制, 跨境/黄金动量第一"
        STATE["ne_fb_raw"] = False
    elif bull and GOLD in info and info[GOLD]["mom20"] > 0:
        target, reason = GOLD, "竞赛池无人达标, 黄金动量为正(备胎)"
        STATE["ne_fb_raw"] = False              # 备胎黄金 MOM20>0, 属合格信号, 不算兜底
    elif strategy.NEVER_EMPTY:
        # v10 钩子 ne_gold_first: 兜底时黄金 MOM20>0 则优先黄金(防御优先于 score 排名)
        if _cfg("ne_gold_first") and GOLD in info and info[GOLD]["mom20"] > 0:
            cand = (GOLD, info[GOLD])
        elif strategy.SAFE_MIN_VOL:
            cands = [x for x in table if x[0] in (set(strategy.GLOBAL_POOL) | {GOLD} | _FBX | _GLX)]
            cand = min(cands, key=lambda x: x[1]["vol"]) if cands else None
        else:
            cand = next((x for x in table
                         if x[0] in (set(strategy.GLOBAL_POOL) | {GOLD} | _FBX | _GLX)), None)
        if strategy.NE_BULL_ONLY and not bull:
            target, reason = CASH, "熊市体制, 空仓避险"
            STATE["ne_fb_raw"] = False
        # v10 钩子 ne_score0: 兜底质量门改 score 口径 —— 最强避险 WLS 斜率仍负则宁可持币
        elif cand and _cfg("ne_score0") and cand[1]["score"] <= 0:
            target, reason = CASH, "避险最强 %s WLS斜率仍负, 持币(不空仓质量门score)" % cand[0]
            STATE["ne_fb_raw"] = False
        elif cand and cand[1]["mom20"] >= strategy.NE_MIN_MOM:
            target, reason = cand[0], "竞赛池无人达标, 避险池%s接盘(永不空仓)" % (
                "最低波动" if strategy.SAFE_MIN_VOL else "最强")
            STATE["ne_fb_raw"] = True     # v10: 兜底候选(包装层按最终返回裁定 ne_fb, 防缓冲拦截误标)
            STATE["ne_fb_cand"] = cand[0]
            STATE["ne_fb_bull"] = bull
        elif cand:
            target, reason = CASH, "避险最强 %s MOM20 %.1f%% 低于兜底门, 持币" % (
                cand[0], cand[1]["mom20"] * 100)
            STATE["ne_fb_raw"] = False
        else:
            target, reason = CASH, "无人达标, 空仓避险" if bull else "熊市体制, 空仓避险"
            STATE["ne_fb_raw"] = False
    else:
        target, reason = CASH, "无人达标, 空仓避险" if bull else "熊市体制, 空仓避险"
        STATE["ne_fb_raw"] = False

    if holding and holding != target and holding in info:
        h = info[holding]
        if strategy.LEAD_EXIT and holding != CASH:
            b300 = info.get(strategy.BULL_CODE, {}).get("mom20", -9)
            best_mom = max((info[c]["mom20"] for c in comp_pool if c in info), default=-9)
            if b300 > best_mom and holding != strategy.BULL_CODE:
                cand = next((x for x in table if x[0] in strategy.GLOBAL_POOL + [GOLD]
                             and x[0] != holding), None)
                tgt = cand[0] if cand else CASH
                return tgt, "领导力背离: 沪深300 MOM20 %+.1f%% 超池内最强 %+.1f%%, 离场" % (
                    b300 * 100, best_mom * 100)
        if not bull and not strategy.BEAR_OPEN_STOCK \
                and holding in (set(strategy.STOCK_POOL) | _STX) and holding not in (_DEF | _BEX):
            # 制度性离场(A股池及stock_extra; defensive/bear_extra 豁免), 优先于滞回缓冲
            return target, "熊市体制确立, A股持仓无条件离场 -> %s" % UNIVERSE[target][0]
        exit_why = _exit_hit_v10(h)
        if exit_why:
            return target, "%s, 离场 -> %s" % (exit_why, UNIVERSE[target][0])
        if strategy.MIN_HOLD and holding_days < strategy.MIN_HOLD and holding != CASH:
            return holding, "持仓满%d天前不轮动(当前%d天), 继续持有" % (strategy.MIN_HOLD, holding_days)
        if strategy.RANK_EXIT_N > 0:
            comp_ranked = [c for c, _ in table if c in comp_pool]
            if holding in comp_ranked and comp_ranked.index(holding) < strategy.RANK_EXIT_N:
                return holding, "排名缓冲: 持仓仍居竞赛池前%d, 继续持有" % strategy.RANK_EXIT_N
        # 滞回缓冲: 挑战者须显著更强
        buf = strategy.BUFFER
        if strategy.POOL_BUFFER and target in info:
            _role = UNIVERSE.get(target, ("", "", ""))[2]
            buf = strategy.POOL_BUFFER.get("gold" if target == GOLD else _role, strategy.BUFFER)
        if strategy.TREND_BUF_ON and h["mom20"] > 0.10:
            buf = max(buf, 0.05)
        if strategy.SLOPE_DAYS_BUF > 0 and h.get("slope_pos_days", 0) >= strategy.SLOPE_DAYS_BUF:
            buf *= 2
        if strategy.VOL_BUFFER > 0 and target in info:
            buf = max(strategy.BUFFER, strategy.VOL_BUFFER * info[target]["vol"] * 20 ** 0.5)
        if strategy.ER_BUFFER_ON and h.get("er20", 1.0) < strategy.KAUFMAN_THR:
            buf *= 2
        if strategy.REL_BUFFER_ON and target in info:
            t_tail, h_tail = info[target].get("rets_tail"), h.get("rets_tail")
            if t_tail and h_tail and len(t_tail) == len(h_tail):
                n = len(t_tail)
                mt = sum(t_tail) / n
                mh = sum(h_tail) / n
                cov = sum((t_tail[i] - mt) * (h_tail[i] - mh) for i in range(n)) / n
                vt = sum((x - mt) ** 2 for x in t_tail) / n
                vh = sum((x - mh) ** 2 for x in h_tail) / n
                corr = cov / (vt * vh) ** 0.5 if vt > 0 and vh > 0 else 0.0
                if corr > 0.85:
                    buf *= 2
        # v10 钩子: 熊市缓冲加倍 / 跨池加罚 / 避险池内缓冲x2
        if _cfg("bearbuf") and not bull:
            buf *= _cfg("bearbuf")
        if _cfg("crosspool") and target in info and holding in UNIVERSE:
            if UNIVERSE[target][2] != UNIVERSE[holding][2]:
                buf += _cfg("crosspool")
        if _cfg("ne_buf2") and target in info:
            safe = strategy.GLOBAL_POOL + [GOLD]
            if holding in safe and target in safe:
                buf *= 2
        # v10 钩子 scorebuf: 缓冲改用排名指标口径(score 相对差), 替代 MOM20 绝对差
        if _cfg("scorebuf") is not None and target in info:
            k = _cfg("scorebuf")
            need = abs(h["score"]) * k
            if info[target]["score"] - h["score"] < need:
                return holding, "滞回缓冲(score): 挑战者(%s score %+.1f) 未超持仓(%s %+.1f) %.0f%%相对差, 继续持有" % (
                    UNIVERSE[target][0], info[target]["score"],
                    UNIVERSE[holding][0], h["score"], k * 100)
        else:
            c_mom = info[target]["mom20"] if target in info else 0.0
            if c_mom - h["mom20"] < buf:
                return holding, "滞回缓冲: 挑战者(%s %+.2f%%) 未超持仓(%s %+.2f%%) %.0f%%, 继续持有" % (
                    UNIVERSE[target][0], c_mom * 100,
                    UNIVERSE[holding][0], h["mom20"] * 100, buf * 100)
        # v10 钩子 rankconfirm: 挑战者须连续 N 天通过缓冲才轮动(治震荡打脸; 敏感离场不受影响)
        if _cfg("rankconfirm", 1) > 1 and target != CASH:
            if STATE["pending"] == target:
                STATE["count"] += 1
            else:
                STATE["pending"], STATE["count"] = target, 1
            if STATE["count"] < _cfg("rankconfirm"):
                STATE["_keep"] = True
                return holding, "排名确认: 挑战者 %s 第%d/%d天通过缓冲, 继续持有" % (
                    UNIVERSE[target][0], STATE["count"], _cfg("rankconfirm"))
            STATE["pending"], STATE["count"] = None, 0
        reason += " (挑战者显著更强, 轮动)"
    return target, reason


def decide_v10(table, holding, holding_days=99):
    """decide_v10 包装: 持仓变更/非挂起路径时重置 rankconfirm 状态;
    ne_fb 最终裁定 = 兜底候选 且 最终返回目标确为该候选(缓冲/确认拦截兜底切换时不算兜底态)。"""
    if holding != STATE.get("holding"):
        STATE["pending"], STATE["count"] = None, 0
    STATE["holding"] = holding
    out = _decide_v10_impl(table, holding, holding_days)
    STATE["ne_fb"] = bool(STATE.get("ne_fb_raw")) and out[0] == STATE.get("ne_fb_cand")
    if not STATE.pop("_keep", False):
        STATE["pending"], STATE["count"] = None, 0
    return out


_orig_decide = strategy.decide


def decide_dispatch(table, holding, holding_days=99):
    if _cfg("use_copies"):
        return decide_v10(table, holding, holding_days)
    return _orig_decide(table, holding, holding_days)


strategy.decide = decide_dispatch


# ---------------------------------------------------------------- 引擎副本(仓位类钩子)
def backtest_v10(histories, calendar, start, end="9999",
                 ne_pos=1.0, ne_pos_bear_only=False, crash_alloc=1.0, crash_pick="score"):
    """复刻 backtest.backtest 主循环(v9.1 默认口径: 费用双边万二/深跌抄底-8%+低年线20%锁5天/
    无熔断无止损无波动目标)。新增钩子:
      ne_pos: 不空仓兜底态(decide 标记 STATE['ne_fb'])期间仓位降为 ne_pos, 余仓货币ETF;
              抄底锁仓期不参与(crash_alloc 管辖)。ne_pos_bear_only: 仅熊市兜底才降仓。
      crash_alloc: 抄底锁仓期仓位(v9.1 引擎原有参数, 默认1.0 从未测过)。
    策略全局参数先经一次极短原始回测调用重置为 v9.1 默认(防抄漏 60 个赋值)。
    保真校验: 变体 engine_check(ne_pos=1.0) 必须逐位等于 baseline。
    """
    backtest.backtest(histories, calendar, start="2024-01-02", end="2024-01-04")
    strategy._state_bull = None
    strategy.BULL_HYST_PENDING = None
    days = [d for d in calendar if start <= d <= end]
    close_of = {c: {r[0]: r[2] for r in rows} for c, rows in histories.items()}
    cash_close = close_of[CASH]

    nav, peak, max_dd = 1.0, 1.0, 0.0
    holding, switches, daily, trades = None, 0, [], []
    holding_days = 0
    pos = 1.0
    lock_until = -1
    crash_buys = []
    for i, d in enumerate(days):
        if i > 0:  # 当日收益按"昨日定下的持仓与仓位"计算
            prev, cur = close_of[holding].get(days[i - 1]), close_of[holding].get(d)
            if prev and cur:
                r_asset = cur / prev - 1.0
                if pos < 1.0:
                    c0, c1 = cash_close.get(days[i - 1]), cash_close.get(d)
                    r_cash = (c1 / c0 - 1.0) if (c0 and c1) else 0.0
                    nav *= 1.0 + pos * r_asset + (1.0 - pos) * r_cash
                else:
                    nav *= cur / prev
            holding_days += 1
        table = strategy.rank(histories, on_date=d)
        target, _ = strategy.decide(table, holding, holding_days)
        ne_fb, ne_fb_bull = STATE.get("ne_fb", False), STATE.get("ne_fb_bull", True)
        # 恐慌抄底(v9 默认口径): 全池 MOM5<=-8% 且低于年线20% -> 买入锁仓5天
        if i < lock_until:
            target = holding
        else:
            crash_pool = backtest.STOCK_POOL + backtest.GLOBAL_POOL + [backtest.GOLD]
            cands = [x for x in table if x[0] in crash_pool
                     and x[1]["mom5"] <= -0.08 and x[0] != holding
                     and x[1]["dist_ma250"] < -0.20]
            if crash_pick == "lowest":
                cand = min(cands, key=lambda x: x[1]["mom5"]) if cands else None
            else:
                cand = cands[0] if cands else None
            if cand:
                target = cand[0]
                lock_until = i + 5
                crash_buys.append((d, cand[0]))
        if target != holding:
            if i > 0:                # 首日建仓不计换手
                switches += 1
                nav *= (1 - backtest.FEE * 2)
                trades.append((d, holding, target, nav))
            holding = target
            holding_days = 0
        pos = 1.0
        # v10 钩子: 抄底锁仓期仓位(crash_alloc), 优先于兜底仓位
        if crash_alloc < 1.0 and i < lock_until and holding != CASH:
            pos = crash_alloc
        elif 0 < ne_pos < 1.0 and holding != CASH and ne_fb and i >= lock_until:
            # i >= lock_until: 锁仓期内 decide 的兜底标记对抄底持仓无效, 不得降仓
            if not (ne_pos_bear_only and ne_fb_bull):
                pos = ne_pos
        peak = max(peak, nav)
        max_dd = min(max_dd, nav / peak - 1)
        daily.append((d, nav, holding))
        if _cfg("trace"):
            TRACE.append((d, holding, pos, ne_fb, ne_fb_bull))

    n = len(daily)
    rets = [daily[i][1] / daily[i - 1][1] - 1 for i in range(1, n)]
    mean = sum(rets) / len(rets)
    std = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
    years = n / backtest.TRADING_DAYS
    ann = nav ** (1 / years) - 1
    return {
        "nav": nav, "ann": ann, "max_dd": max_dd,
        "sharpe": mean / std * backtest.TRADING_DAYS ** 0.5 if std else 0,
        "calmar": ann / abs(max_dd) if max_dd else 0,
        "switches": switches, "sw_per_year": switches / years,
        "daily": daily, "trades": trades, "crash_buys": crash_buys,
    }


# ---------------------------------------------------------------- 变体注册表
VARIANTS = {
    # 校验
    "baseline":      ({}, {}),
    "copy_check":    ({"use_copies": True}, {}),
    # 不空仓簇(重点方向): kwarg 类
    "ne_gate_m5":    ({}, {"ne_min_mom": -0.05}),
    "ne_gate_m10":   ({}, {"ne_min_mom": -0.10}),
    "ne_minvol":     ({}, {"safe_min_vol": True}),
    "ne_bull_only":  ({}, {"ne_bull_only": True}),
    # 不空仓簇: 副本钩子类
    "ne_score0":     ({"use_copies": True, "ne_score0": True}, {}),
    "ne_gold_first": ({"use_copies": True, "ne_gold_first": True}, {}),
    "ne_buf2":       ({"use_copies": True, "ne_buf2": True}, {}),
    # 缓冲/排名簇
    "scorebuf_010":  ({"use_copies": True, "scorebuf": 0.10}, {}),
    "scorebuf_015":  ({"use_copies": True, "scorebuf": 0.15}, {}),
    "scorebuf_020":  ({"use_copies": True, "scorebuf": 0.20}, {}),
    "scorebuf_030":  ({"use_copies": True, "scorebuf": 0.30}, {}),
    "rankconfirm2":  ({"use_copies": True, "rankconfirm": 2}, {}),
    "rankconfirm3":  ({"use_copies": True, "rankconfirm": 3}, {}),
    "bearbuf_15":    ({"use_copies": True, "bearbuf": 1.5}, {}),
    "bearbuf_2":     ({"use_copies": True, "bearbuf": 2.0}, {}),
    "crosspool_001": ({"use_copies": True, "crosspool": 0.01}, {}),
    "crosspool_002": ({"use_copies": True, "crosspool": 0.02}, {}),
    # 进出场合一簇
    "enter_slope":   ({"use_copies": True, "enter_slope": True}, {}),
    "exit_slope":    ({"use_copies": True, "exit_slope": True}, {}),
    "slope_both":    ({"use_copies": True, "enter_slope": True, "exit_slope": True}, {}),
    # 其他结构候选
    "gold_exempt":   ({"use_copies": True, "gold_exempt": True}, {}),
    "wls2030":       ({"wls2030": True}, {}),
    # 廉价 kwarg 在 v9.1 基线上重测
    "sortino_wls":   ({}, {"use_downside_vol": True}),
    "crash_lowest":  ({}, {"crash_pick": "lowest"}),
    "oh_lookback3":  ({}, {"overheat_lookback": 3}),
    "score_smooth3": ({}, {"score_smooth": 3}),
    # ---- 第二轮: 不空仓仓位管理 + 抄底仓位(引擎副本钩子) ----
    "engine_check":   ({"use_copies": True}, {"engine": True}),
    "ne_half70":      ({"use_copies": True}, {"engine": True, "ne_pos": 0.7}),
    "ne_half50":      ({"use_copies": True}, {"engine": True, "ne_pos": 0.5}),
    "ne_half30":      ({"use_copies": True}, {"engine": True, "ne_pos": 0.3}),
    "ne_half50_bear": ({"use_copies": True}, {"engine": True, "ne_pos": 0.5, "ne_pos_bear_only": True}),
    "ne_s0_half50":   ({"use_copies": True, "ne_score0": True}, {"engine": True, "ne_pos": 0.5}),
    "crash_alloc70":  ({"use_copies": True}, {"engine": True, "crash_alloc": 0.7}),
    "crash_alloc50":  ({"use_copies": True}, {"engine": True, "crash_alloc": 0.5}),
    "ne50_ca70":      ({"use_copies": True}, {"engine": True, "ne_pos": 0.5, "crash_alloc": 0.7}),
    # ---- 第三轮: 红利低波角色/新增标的/次方量化借鉴(2026-08-29) ----
    # 512890 红利低波(已在A股池) 防御化组合
    "dv_fb":          ({"use_copies": True, "fb_extra": ["512890"]}, {}),
    "dv_def":         ({"use_copies": True, "defensive": ["512890"]}, {}),
    "dv_def_fb":      ({"use_copies": True, "defensive": ["512890"], "fb_extra": ["512890"]}, {}),
    "dv_def_ng":      ({"use_copies": True, "defensive": ["512890"], "bear_gate_exempt": ["512890"]}, {}),
    "dv_def_fb_ng":   ({"use_copies": True, "defensive": ["512890"], "fb_extra": ["512890"],
                        "bear_gate_exempt": ["512890"]}, {}),
    # 510880 上证红利(2010+数据) 各种角色
    "add880_stock":   ({"use_copies": True, "stock_extra": ["510880"]}, {}),
    "add880_stock_fb":({"use_copies": True, "stock_extra": ["510880"], "fb_extra": ["510880"]}, {}),
    "add880_def":     ({"use_copies": True, "defensive": ["510880"]}, {}),
    "add880_def_fb":  ({"use_copies": True, "defensive": ["510880"], "fb_extra": ["510880"]}, {}),
    # 双红利防御组合
    "combo_def_fb":   ({"use_copies": True, "defensive": ["512890", "510880"],
                        "fb_extra": ["512890", "510880"]}, {}),
    # 短历史红利(2019末/2020中上市, 仅参考)
    "add080_stock":   ({"use_copies": True, "stock_extra": ["515080"]}, {}),
    "add100_stock":   ({"use_copies": True, "stock_extra": ["515100"]}, {}),
    # 国债防御层(平台对照; bear_extra=仅熊市参赛, fb=兜底)
    "bond_fb":        ({"use_copies": True, "fb_extra": ["511010"]}, {}),
    "bond_bear_fb":   ({"use_copies": True, "bear_extra": ["511010"], "fb_extra": ["511010"]}, {}),
    "bond_def_fb":    ({"use_copies": True, "defensive": ["511010"], "fb_extra": ["511010"]}, {}),
    # 次方量化借鉴: score=slope×R²(全强度) / 原式(对数价, 权重1→2, 不除vol)
    "r2full":         ({"score_r2_full": True}, {}),
    "r2_cifang":      ({"score_cifang": True}, {}),
    # 次方量化池结构借鉴: 德国/日经/豆粕 进跨境池(牛熊参赛+兜底+3%缓冲)
    "gx_de":          ({"use_copies": True, "global_extra": ["513030"]}, {}),
    "gx_jp":          ({"use_copies": True, "global_extra": ["513520"]}, {}),
    "gx_dp":          ({"use_copies": True, "global_extra": ["159985"]}, {}),
    "gx_all3":        ({"use_copies": True, "global_extra": ["513030", "513520", "159985"]}, {}),
}

STARTS = ("2014-01-01", "2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01")


def run_variant(name, start):
    """跑单个变体×起点, 返回指标 dict。每 run 重置 CFG/STATE; strategy 全局由引擎重置。"""
    init_data()
    cfg, kw = VARIANTS[name]
    CFG.clear()
    CFG.update(cfg)
    STATE.update(last_date=None, holding=None, pending=None, count=0,
                 ne_fb=False, ne_fb_bull=True, ne_fb_raw=False, ne_fb_cand=None)
    STATE.pop("_keep", None)
    kw = dict(kw)
    if kw.pop("engine", False):
        r = backtest_v10(H, CAL, start=start, **kw)
    else:
        r = backtest.backtest(H, CAL, start=start, **kw)
    yearly = dict(backtest.yearly(r["daily"]))
    return {
        "variant": name, "start": start,
        "nav": round(r["nav"], 4), "ann": round(r["ann"] * 100, 2),
        "dd": round(r["max_dd"] * 100, 2), "sharpe": round(r["sharpe"], 3),
        "calmar": round(r["calmar"], 2), "switches": r["switches"],
        "sw_per_year": round(r["sw_per_year"], 1),
        "crash_buys": len(r["crash_buys"]),
        "yearly": {y: round(x * 100, 1) for y, x in yearly.items()},
    }


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["baseline"]
    starts = sys.argv[2].split(",") if len(sys.argv) > 2 else list(STARTS[:2])
    for nm in names:
        for st in starts:
            try:
                print(json.dumps(run_variant(nm, st), ensure_ascii=False), flush=True)
            except Exception as e:
                print(json.dumps({"variant": nm, "start": st, "error": repr(e)},
                                 ensure_ascii=False), flush=True)
