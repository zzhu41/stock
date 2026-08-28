# -*- coding: utf-8 -*-
"""动量轮动策略核心 v9.1：牛熊开关 + WLS25 斜率排名 + 分池缓冲 + 深跌抄底。

两层结构, 各司其职:
  体制层: 沪深300ETF 跌破年线(MA250) = 熊市体制, A股池禁止持有, 只允许跨境/黄金/货币;
          站上 = 牛市体制, 股池正常轮动。专治阴跌市假反弹(2018)与持续熊市(2022)。
  动量层: score = 25日时间加权回归斜率/VOL20 排名(v8 转正); 进场 MOM20>0(熊市门槛7%, v8.1);
          持仓 MOM20 转负敏感离场; 轮动需挑战者 MOM20 超持仓分池缓冲(A股2%/跨境3%/黄金3%, v9.1)。

存活组件(15轮250+变体验证, 详见 REPRODUCE.md):
  急跌离场(v6): 持仓单日跌≤-4% 当日离场, 左尾保险;
  永不空仓(v7): 竞赛池无人达标时持避险池最强者(跨境+黄金), 不蹲货币;
  过热快离场: MOM20>40% 时离场收紧为 MOM5<0(防抛物线顶, 如2024-10);
  深跌恐慌抄底(v9): MOM5≤-8% 且低于年线20% → 买入锁仓5交易日(危机Alpha, 信号端锁仓状态存 signals/crash_lock.json)。
以下试验开关均为证伪存档, 默认关闭, 勿打开(打开即偏离论文口径)。
"""
from market_data import UNIVERSE, STOCK_POOL, GLOBAL_POOL, GOLD, CASH

MOM_WINDOWS = (5, 20, 60)
MOM_MAIN = 20           # 主动量窗口(进场/离场/缓冲/排名分子); 过热判断同步
VOL_DAYS = 20
MA_BULL = 250           # 牛熊开关均线
BULL_CODE = "510300"    # 单基准模式
BULL_VOTE = False       # True=多数表决: 股票ETF≥2只在年线上才为牛市
BUFFER = 0.02
OVERHEAT = 0.40         # MOM20 超过此值进入过热态
OVERHEAT_LOOKBACK = 1   # 过热判断回看窗口: max(MOM20 近L日)>OVERHEAT 即视为过热; 1=仅当日(原行为)
PANIC_DROP = 0.04       # 持仓单日跌幅超此值触发紧急离场(次日若仍动量第一可立即买回); 0=关闭
                        # v6 新增(2026-08-18验证): 全起点/逐年稳健, 本质是左尾保险而非阿尔法
GLOBAL_MA_FILTER = False    # True=跨境池进场须站上自身年线(MA250)
GLOBAL_MOM60_FILTER = False # True=跨境池进场须 MOM60>0
# --- 文献试验参数(默认关闭, 2026-08-18 第二轮) ---
POS_FRAC_MIN = 0.0      # Frog-in-the-Pan(Da et al 2014): 进场要求近20日上涨天数占比>=此值; 0=关闭
MAX_CAP = 9.9           # MAX彩票效应(Bali et al 2011): 近20日最大单日涨幅>此值禁止进场; 9.9=关闭
USE_DOWNSIDE_VOL = False  # True=score用下行波动率做分母(Sortino)
# --- 结构试验参数(默认关闭, 第三轮) ---
DUAL_MOM = False        # True=进场须 MOM20>0 且 MOM60>0(双动量确认)
SCORE_MOM = 0           # 排名分子窗口: 0=用主动量; 60=用MOM60(慢动量排名)
MA_SLOPE = False        # True=牛市除站上年线外, 还须年线本身上行(MA斜率>0, 20日)
RANK_EXIT_N = 0         # >0时启用排名缓冲: 持仓排名<=N且未触离场线则不轮动(替代MOM差值缓冲)
SCORE_PLAIN_MOM = False  # True=纯动量排名(mom20不除波动率), 更激进: 永远咬涨最猛的
NEVER_EMPTY = True      # v7 新增(2026-08-19验证): 竞赛池无人达标时不空仓, 持避险池(跨境+黄金)score最强
                        # 6/6起点终值全赢, 2014起年化+39.4% vs +35.3%; 代价: 回撤-28.1%, 换手x4
# --- 技术指标试验参数(默认关闭, 第六轮 2026-08-19) ---
MACD_FILTER = False     # 进场须 MACD 金叉状态(DIF>DEA)
MACD_EXIT = False       # MACD 死叉(DIF<DEA)离场
KDJ_NOCHASE = False     # J>100(超买)禁止进场/切换(不追高)
KDJ_DIP_BUY = False     # J<10(超卖)时放宽进场: MOM20>-2%即可(抄反弹)
RSI_NOCHASE = False     # RSI14>80 禁止进场/切换
MA_ALIGN = False        # 多头排列才进场: close>MA20 且 MA5>MA10>MA20
# --- 第七轮穷举扩展(默认关闭) ---
MACD_DIF_POS = False    # 进场须 DIF>0(零轴上方)
KDJ_GOLDEN = False      # 进场须 K>D(KDJ金叉状态)
KDJ_DEAD_EXIT = False   # K<D(KDJ死叉)离场
RSI_GT50 = False        # 进场须 RSI14>50(强弱分水岭)
RSI_LT50_EXIT = False   # RSI14<50 离场
ABOVE_MA20 = False      # 进场须 close>MA20
MA20_RISING = False     # 进场须 MA20 上行(MA20>5日前MA20)
BIAS_NOCHASE = 9.9      # BIAS20=(close/MA20-1)>此值禁追高; 9.9=关闭
PCTB_NOCHASE = 9.9      # 布林%B>此值禁追高; 9.9=关闭
CCI_NOCHASE = 9999.0    # CCI14>此值禁追高; 9999=关闭
KDJ_NC_THR = 100.0      # KDJ禁追高阈值(配合KDJ_NOCHASE)
RSI_NC_THR = 80.0       # RSI禁追高阈值(配合RSI_NOCHASE)
SCORE_TECH_MIX = False  # 打分混合: score = (mom20/vol + macd柱占比/vol)/2
# --- 止损专项(默认关闭, 第八轮 2026-08-19, 针对2022全负动量年) ---
NE_MIN_MOM = -9.9       # 永不空仓兜底质量门: 最强避险MOM20低于此值宁可持货币; -9.9=关闭
NE_BULL_ONLY = False    # True=永不空仓仅牛市生效, 熊市回到货币兜底
EXIT_MOM_FLOOR = 0.0    # 离场动量阈值: MOM20<=此值离场(默认0=转负); 正值=提前离场
MOM_DECAY = 9.9         # 动量衰减止损: 近20日MOM20峰值-当前MOM20>此值离场; 9.9=关闭
DONCHIAN_N = 0          # 唐奇安低点止损: 收盘跌破近N日最低收盘离场; 0=关闭
# --- v8 方向参数(默认关闭, 第九轮 2026-08-21) ---
ER_BUFFER_ON = False    # 震荡市自适应缓冲: 持仓ER20<KAUFMAN_THR时缓冲x2
KAUFMAN_THR = 0.30      # 效率系数阈值(|净位移|/路径长度, 低于=震荡)
REL_BUFFER_ON = False   # 同因子防切: 挑战者与持仓近60日相关系数>0.85时缓冲x2
VOL_PANIC_ON = False    # panic阈值按标的vol缩放: clip(2*vol20, 3%, 6%) 替代固定4%
VOL_PANIC_K = 2.0
VOL_PANIC_LO = 0.03
VOL_PANIC_HI = 0.06
W_MOM = False           # True=线性加权动量(近20日逐日收益按时间加权, 近期权重高)替代首尾MOM20做score
VOL_RATIO_MAX = 0.0     # >0时启用: vol5/vol20>此值(波动飙升)禁止进场
# --- 激进组合参数(默认关闭, 第十轮 2026-08-21) ---
ENTER_MOM_MIN = 0.0     # 进场动量阈值: MOM20须>此值才进场(更强趋势确认)
SCORE_SQUARE = False    # True=score用 mom20*|mom20|/vol (放大强者差异, 激进追强)
SCORE_WLS = True        # v8 核心(2026-08-21转正): 25日时间加权回归斜率/vol 做score(次方量化思路)
                        # 验证: 6/6起点全赢, 23-28窗口连续平台, 2014起年化+41.1%/夏普1.49/终值+7521%
                        # 代价: 逐年波动大(2021 -16.6pp/2023 -18.8pp 单年大输基线)
# --- 成交量维度(默认关闭, 第十轮) ---
VOL_IN_MIN = 0.0        # 进场量比门槛: 当日量/20日均量 >= 此值(放量确认); 0=关闭
VOL_IN_MAX = 99.0       # 天量禁追: 量比>此值禁止进场; 99=关闭
VOL_SCORE_MIX = False   # True=score按量比加权: score *= (0.5+0.5*min(量比,2))
# --- 回撤控制参数(默认关闭, 第十一轮 2026-08-21) ---
SCORE_SMOOTH = 1        # score平滑: 近N日score均值(降噪); 1=不平滑
ENTER_VOL_MAX = 9.9     # 高波禁进场: 标的年化vol>此值禁止新进场; 9.9=关闭
BEAR_ENTER_MOM = 0.07   # v8.1(2026-08-21转正): 熊市进场门槛MOM20>7%(过滤熊市弱反弹, 减少打脸)
                        # 验证: 6/6起点终值全赢, 门槛5-10%连续平台; 2014起+42.2%/+8334%, 回撤不变
                        # 代价: 2022年-4.3pp(熊市真反弹也被滤掉部分)
BEAR_OPEN_STOCK = False  # True=熊市竞赛池含A股(取消禁入, 无制度性离场, 仅靠门槛把关)
# --- v9 候选参数(默认关闭, 第十二轮 2026-08-21) ---
SCORE_R2 = False        # True=score乘趋势质量: (0.5+0.5*R²), R²=WLS回归拟合优度(趋势直=高分)
SCORE_TSTAT = False     # True=score用WLS斜率的t统计量(slope/标准误, 统计显著趋势)替代slope/vol
WLS60_MIX = 0.0         # >0时score混入WLS60斜率: (1-w)*WLS25/vol + w*WLS60/vol
VOL_EWM = False         # True=波动率用EWM(halflife10, 近期权重高)替代等权vol20
BULL_DUAL = False       # True=牛市须双确认: 站上年线 且 沪深300自身MOM20>0
# --- 平台移植试验(默认关闭, 第十四轮 2026-08-25) ---
LEAD_EXIT = False       # 领导力背离: 沪深300的mom20>竞赛池最强mom20 -> 持仓离场去避险(平台"上证空仓"原型)
ZSCORE_CRASH = False    # 抄底触发改z-score: mom10<μ-2.5σ(自适应)替代固定-8%
ZSCORE_ACCEL = False    # 加速止盈: mom10>μ+3σ -> 离场(z-score过热, 替代固定40%门)
# --- v10 第二轮(默认关闭, 2026-08-25) ---
BULL_HYST = 0.0         # 牛熊滞后带: 跌破年线(1+band)才算熊/站回(1+band)才算牛; 0=关闭
BULL_CONFIRM = 1        # 体制切换确认天数: 翻转需连续N天才生效; 1=关闭
_state_bull = None      # 滞后带/确认用的体制状态(backtest每次调用重置)
BULL_HYST_PENDING = None
WLS_ADAPTIVE = False    # True=WLS窗口按vol自适应: 年化vol>30%用30日, 否则20日
RESID_PENALTY = False   # True=回归线乖离减分: 价偏离WLS趋势线>2σ时score减半(防追高)
# --- 借鉴平台机制(默认关闭, 2026-08-25) ---
TREND_BUF_ON = False    # 趋势强度自适应缓冲: 持仓MOM20>10%%(强趋势)时缓冲升至5%%, 不轻易下车
POOL_BUFFER = {"stock": 0.02, "global": 0.03, "gold": 0.03}  # 分池缓冲(v9.1, 2026-08-25转正):
                        # A股2%/跨境3%(QDII趋势持续+溢价噪声, 防甩下车)/黄金3%(咬避险趋势)
                        # 验证: 6/6起点全赢, 逐年7年不变+2021/2024/2026各+0.9~7.6pp; 代价2025-4.3pp
SLOPE_DAYS_BUF = 0      # 斜率连续为正确认: WLS斜率连续N日为正时缓冲x2; 0=关闭
SAFE_MIN_VOL = False    # 保守兜底: 无人达标时持避险池中vol最低者(而非score最强)
VOL_BUFFER = 0.0        # ATR缩放缓冲: buf=K*vol20*sqrt(20), 替代固定2%; 0=关闭


def _need_tech():
    """任一技术指标开关开启才计算指标(回测开销控制), 运行时判断。"""
    return (MACD_FILTER or MACD_EXIT or KDJ_NOCHASE or KDJ_DIP_BUY or RSI_NOCHASE
            or MA_ALIGN or MACD_DIF_POS or KDJ_GOLDEN or KDJ_DEAD_EXIT or RSI_GT50
            or RSI_LT50_EXIT or ABOVE_MA20 or MA20_RISING or BIAS_NOCHASE < 9
            or PCTB_NOCHASE < 9 or CCI_NOCHASE < 999 or SCORE_TECH_MIX)
CIRCUIT_DD = 0.0        # 组合净值回撤熔断阈值, 0=关闭; 触发后冷却 CIRCUIT_DAYS 天
CIRCUIT_DAYS = 10
SKIP_DAYS = 0           # 动量计算跳过最近N日(短期反转过滤, Jegadeesh1990)
USE_HIGH52 = False      # True=用52周高点接近度排名(George&Hwang2004)
MIN_HOLD = 0            # 轮动最小持仓天数(敏感离场/熊市离场不受限)
MIN_ROWS = max(MA_BULL, max(MOM_WINDOWS) + 1 + SKIP_DAYS) + VOL_DAYS


def _ema_series(vals, n):
    """标准 EMA 序列(种子=首值, 长历史下初期误差可忽略)。"""
    alpha = 2.0 / (n + 1)
    out = [vals[0]]
    for v in vals[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def indicators(closes, volumes=None):
    """closes 升序; volumes 可选(与 closes 等长, 量比指标用)。返回指标 dict 或 None。

    SKIP_DAYS>0 时 mom20/mom60 跳过最近N日计算(短期反转过滤);
    mom5 不跳过(过热快离场需要最敏感); USE_HIGH52 时 score 改为52周高点接近度。
    技术指标(默认参数关闭时才不算, 为省开销由调用方保证窗口足够):
      MACD(12,26,9) dif/dea; RSI14(Wilder); KDJ 的 J(close极值近似, 无high/low数据);
      ma_align: close>MA20 且 MA5>MA10>MA20。
    """
    if len(closes) < MIN_ROWS:
        return None
    e = 1 + SKIP_DAYS
    w = MOM_MAIN
    m = {5: closes[-1] / closes[-6] - 1.0,
         w: closes[-1] / closes[-w - 1] - 1.0,           # 主动量, 用于进场/离场/缓冲(要快)
         60: closes[-1] / closes[-61] - 1.0}
    m20_rank = closes[-e] / closes[-e - w] - 1.0 if SKIP_DAYS else m[w]  # 仅排名用
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(len(closes) - VOL_DAYS, len(closes))]
    mean = sum(rets) / len(rets)
    vol = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
    if VOL_EWM:  # 指数加权波动率(halflife=10, 近期权重高)
        lam = 0.5 ** (1.0 / 10)
        v = 0.0
        wsum = 0.0
        for r in rets:
            v = lam * v + (1 - lam) * (r ** 2)
            wsum = lam * wsum + (1 - lam)
        vol = (v / wsum) ** 0.5 if wsum > 0 else vol
    dvol = (sum(min(r, 0.0) ** 2 for r in rets) / len(rets)) ** 0.5  # 下行波动率(Sortino分母)
    pos_frac = sum(1 for r in rets if r > 0) / len(rets)              # 上涨天数占比(FIP)
    max_ret = max(rets)                                               # 近20日最大单日涨幅(MAX效应)
    above_ma = closes[-1] > sum(closes[-MA_BULL:]) / MA_BULL
    dist_ma250 = closes[-1] / (sum(closes[-MA_BULL:]) / MA_BULL) - 1.0  # 距年线距离(抄底深跌约束用)
    if MA_SLOPE:  # 年线斜率: 当前MA vs 20日前MA
        ma_now = sum(closes[-MA_BULL:]) / MA_BULL
        ma_prev = sum(closes[-MA_BULL - 20:-20]) / MA_BULL
        ma_rising = ma_now > ma_prev
    else:
        ma_rising = True
    high52 = closes[-1] / max(closes[-244:])
    vol_den = dvol if USE_DOWNSIDE_VOL else vol
    mom_for_score = m[60] if SCORE_MOM == 60 else m20_rank
    ret1 = closes[-1] / closes[-2] - 1.0
    if OVERHEAT_LOOKBACK > 1:  # 近L日最高主动量: 有状态化的过热判断, 防崩盘时动量速降导致保护失效
        mom20_max = max(closes[-k] / closes[-k - MOM_MAIN] - 1.0
                        for k in range(1, OVERHEAT_LOOKBACK + 1))
    else:
        mom20_max = m[MOM_MAIN]
    if MOM_DECAY < 9:  # 近20日主动量峰值(动量衰减止损用)
        mom20_peak20 = max(closes[-k] / closes[-k - MOM_MAIN] - 1.0
                           for k in range(1, 21))
    else:
        mom20_peak20 = mom20_max
    if DONCHIAN_N > 0:   # 前N日最低收盘(唐奇安止损, 不含今日)
        donchian_low = min(closes[-DONCHIAN_N - 1:-1])
    else:
        donchian_low = 0.0
    above_donchian = closes[-1] >= donchian_low
    er20 = 1.0
    rets_tail = None
    if ER_BUFFER_ON or REL_BUFFER_ON:
        seg = closes[-21:]
        path = sum(abs(seg[i] - seg[i - 1]) for i in range(1, len(seg)))
        er20 = abs(seg[-1] - seg[0]) / path if path > 0 else 1.0  # Kaufman效率系数
    if REL_BUFFER_ON:
        rets_tail = [closes[i] / closes[i - 1] - 1.0 for i in range(len(closes) - 60, len(closes))]
    # z-score(平台移植): mom10 相对自身250日历史的 z 值
    mom10_z = 0.0
    if ZSCORE_CRASH or ZSCORE_ACCEL:
        m10s = [closes[i] / closes[i - 10] - 1.0 for i in range(len(closes) - 250, len(closes))]
        mm = sum(m10s) / len(m10s)
        ms = (sum((x - mm) ** 2 for x in m10s) / len(m10s)) ** 0.5
        mom10_z = (m10s[-1] - mm) / ms if ms > 0 else 0.0
    # --- 技术指标(仅相关开关开启时计算, 节省回测开销; close-only 近似, 无high/low) ---
    dif = dea = j_val = rsi = 0.0
    k_val = d_val = 50.0
    ma_align = True
    above_ma20 = ma20_rising = True
    bias20 = pctb = cci = 0.0
    hist_pct = 0.0
    if _need_tech():
        # MACD(12,26,9)
        ema12 = _ema_series(closes, 12)
        ema26 = _ema_series(closes, 26)
        dif_s = [a - b for a, b in zip(ema12, ema26)]
        dea_s = _ema_series(dif_s, 9)
        dif, dea = dif_s[-1], dea_s[-1]
        hist_pct = (dif - dea) / closes[-1]
        # KDJ(9,3,3): K=SMA(RSV,3,1), D=SMA(K,3,1), J=3K-2D
        k_val = d_val = 50.0
        for i in range(max(0, len(closes) - 60), len(closes)):
            win = closes[max(0, i - 8):i + 1]
            lo_i, hi_i = min(win), max(win)
            rsv_i = (closes[i] - lo_i) / (hi_i - lo_i) * 100 if hi_i > lo_i else 50.0
            k_val = k_val * 2 / 3 + rsv_i / 3
            d_val = d_val * 2 / 3 + k_val / 3
        j_val = 3 * k_val - 2 * d_val
        # RSI14(简单均值口径)
        gains = losses = 0.0
        for i in range(len(closes) - 14, len(closes)):
            chg = closes[i] - closes[i - 1]
            gains += max(chg, 0.0)
            losses += max(-chg, 0.0)
        ag, al = gains / 14.0, losses / 14.0
        rsi = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
        # 均线族 / BIAS / %B / CCI14
        ma5 = sum(closes[-5:]) / 5
        ma10 = sum(closes[-10:]) / 10
        ma20 = sum(closes[-20:]) / 20
        ma20_prev = sum(closes[-25:-5]) / 20
        ma_align = closes[-1] > ma20 and ma5 > ma10 > ma20
        above_ma20 = closes[-1] > ma20
        ma20_rising = ma20 > ma20_prev
        bias20 = closes[-1] / ma20 - 1.0
        var20 = sum((x - ma20) ** 2 for x in closes[-20:]) / 20
        sd20 = var20 ** 0.5
        pctb = (closes[-1] - (ma20 - 2 * sd20)) / (4 * sd20) if sd20 > 0 else 0.5
        tp14 = closes[-14:]
        tp_mean = sum(tp14) / 14
        md = sum(abs(x - tp_mean) for x in tp14) / 14
        cci = (closes[-1] - tp_mean) / (0.015 * md) if md > 0 else 0.0
    # 成交量维度: 量比 = 当日量 / 近20日均量(最后一根盘中/昨日量, 近似可用)
    vol_in_ratio = 1.0
    if volumes and len(volumes) == len(closes) and (VOL_IN_MIN > 0 or VOL_IN_MAX < 99 or VOL_SCORE_MIX):
        avg20 = sum(volumes[-21:-1]) / 20.0
        vol_in_ratio = volumes[-1] / avg20 if avg20 > 0 else 1.0
    # score 最终计算(tech混合需要hist_pct, 放指标段之后)
    if SCORE_TECH_MIX:
        score = (mom_for_score / vol_den + hist_pct / vol_den) / 2 if vol_den > 0 else 0.0
    elif SCORE_WLS:  # 25日时间加权回归斜率(次方量化同款): 斜率(百分比/日)/vol
        def _wls(seg):
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
            # t统计量: slope/SE, SE=残差std/sqrt(wsxx); 残差=seg-拟合值
            fitted = [wmy + (wsxy / wsxx) * (xs[i] - wmx) if wsxx > 0 else wmy for i in xs]
            resid = [seg[i] - fitted[i] for i in xs]
            res_std = (sum(r ** 2 for r in resid) / max(n - 2, 1)) ** 0.5
            se = res_std / (wsxx ** 0.5) if wsxx > 0 else 1e9
            t_stat = (wsxy / wsxx) / se if se > 0 else 0.0
            resid_z = resid[-1] / res_std if res_std > 0 else 0.0
            return sl * 250, r2, t_stat, resid_z
        _wls_window = 25
        if WLS_ADAPTIVE and vol > 0:  # 波动自适应窗口: 年化vol>30%用30日窗, 否则20日
            _wls_window = 30 if vol * 244 ** 0.5 > 0.30 else 20
        slope_ann, r2_25, t25, resid_z25 = _wls(closes[-_wls_window:])
        if SCORE_SMOOTH > 1:  # 近N日斜率均值(信号平滑)
            slopes = []
            for k in range(1, SCORE_SMOOTH + 1):
                seg_k = closes[-(25 + k - 1):-(k - 1)] if k > 1 else closes[-25:]
                if len(seg_k) < 25:
                    continue
                sk, _, _, _ = _wls(seg_k)
                slopes.append(sk)
            if slopes:
                slope_ann = sum(slopes) / len(slopes)
        if WLS60_MIX > 0:  # 混入60日WLS(跨周期集成)
            s60, _, _, _ = _wls(closes[-60:])
            slope_ann = (1 - WLS60_MIX) * slope_ann + WLS60_MIX * s60
        if SCORE_TSTAT:
            score = t25                       # t统计量: 统计上显著的趋势(自带量纲)
        else:
            score = slope_ann / vol_den if vol_den > 0 else 0.0
        if SCORE_R2:  # 趋势质量: R²低(噪声走势)的动量打折
            score *= 0.5 + 0.5 * r2_25
        if RESID_PENALTY and abs(resid_z25) > 2.0:  # 回归线乖离>2σ: score减半(防追高)
            score *= 0.5
    elif SCORE_SQUARE:  # mom20^2 保号/vol: 放大强者差异
        sq = mom_for_score * abs(mom_for_score)
        score = sq / vol_den if vol_den > 0 else 0.0
    elif VOL_SCORE_MIX:  # 量比加权: 放量上涨的动量加分
        base = mom_for_score / vol_den if vol_den > 0 else 0.0
        score = base * (0.5 + 0.5 * min(vol_in_ratio, 2.0))
    elif W_MOM:  # 线性加权动量: 近20日逐日收益按时间加权(越近权重越大), 对拐点更敏感
        seg = closes[-21:]
        wr = [(i + 1) * (seg[i + 1] / seg[i] - 1.0) for i in range(20)]
        wmom = sum(wr) * 2.0 / (20 * 21) * 20  # 归一到20日动量量纲
        score = wmom / vol_den if vol_den > 0 else 0.0
    elif SCORE_PLAIN_MOM:
        score = mom_for_score
    else:
        score = high52 if USE_HIGH52 else (mom_for_score / vol_den if vol_den > 0 else 0.0)
    vol5 = 0.0
    if VOL_RATIO_MAX > 0:
        r5 = [closes[i] / closes[i - 1] - 1.0 for i in range(len(closes) - 5, len(closes))]
        m5 = sum(r5) / 5
        vol5 = (sum((r - m5) ** 2 for r in r5) / 5) ** 0.5
    vol_ratio = vol5 / vol if vol > 0 else 0.0
    # 斜率连续为正日数(咬定确认): 近12日每日WLS25斜率符号连续为正的天数
    slope_pos_days = 0
    if SLOPE_DAYS_BUF > 0:
        def _slope_sign(seg):
            n = len(seg)
            xs = list(range(n))
            wts = [i + 1 for i in xs]
            wsum = sum(wts)
            wmx = sum(wts[i] * xs[i] for i in xs) / wsum
            wmy = sum(wts[i] * seg[i] for i in xs) / wsum
            wsxy = sum(wts[i] * (xs[i] - wmx) * (seg[i] - wmy) for i in xs)
            wsxx = sum(wts[i] * (xs[i] - wmx) ** 2 for i in xs)
            return 1 if wsxx > 0 and wsxy > 0 else 0
        for k in range(1, 13):
            seg_k = closes[-(25 + k - 1):-(k - 1)] if k > 1 else closes[-25:]
            if len(seg_k) < 25:
                break
            if _slope_sign(seg_k):
                slope_pos_days += 1
            else:
                break
    return {"mom5": m[5], "mom20": m[MOM_MAIN], "mom60": m[60], "vol": vol,
            "score": score, "above_ma": above_ma, "ret1": ret1, "mom20_max": mom20_max,
            "pos_frac": pos_frac, "max_ret": max_ret, "dvol": dvol, "ma_rising": ma_rising,
            "dif": dif, "dea": dea, "j_val": j_val, "rsi": rsi, "ma_align": ma_align,
            "k_val": k_val, "d_val": d_val, "above_ma20": above_ma20,
            "ma20_rising": ma20_rising, "bias20": bias20, "pctb": pctb, "cci": cci,
            "hist_pct": hist_pct, "mom20_peak20": mom20_peak20, "donchian_low": donchian_low,
            "above_donchian": above_donchian, "er20": er20, "rets_tail": rets_tail,
            "vol_ratio": vol_ratio, "vol_in_ratio": vol_in_ratio, "dist_ma250": dist_ma250,
            "mom10_z": mom10_z, "slope_pos_days": slope_pos_days}


def rank(histories, on_date=None, live_prices=None):
    """返回 [(code, ind_dict), ...] 按 score 降序；含黄金，不含货币。

    live_prices: 盘中实时价 {code: price}，有则替换最后一根收盘价(尾盘场景)。
    """
    table = []
    for code, rows in histories.items():
        if code == CASH:
            continue
        rows = [r for r in rows if (on_date is None or r[0] <= on_date)]
        closes = [r[2] for r in rows]
        volumes = [r[3] for r in rows] if rows and len(rows[0]) > 3 else None
        if live_prices and code in live_prices and closes:
            closes[-1] = live_prices[code]
        ind = indicators(closes, volumes)
        if ind:
            table.append((code, ind))
    table.sort(key=lambda t: t[1]["score"], reverse=True)
    return table


def _enter_ok(ind, code=None, bull=True):
    """进场条件: MOM20>阈值(熊市可用BEAR_ENTER_MOM加门槛); 过热确认; 各过滤器。"""
    mom_floor = -0.02 if (KDJ_DIP_BUY and ind["j_val"] < 10) else ENTER_MOM_MIN
    if not bull and BEAR_ENTER_MOM > mom_floor:
        mom_floor = BEAR_ENTER_MOM
    if ind["mom20"] <= mom_floor:
        return False
    if ind["mom20"] > OVERHEAT and ind["mom5"] <= 0:
        return False
    if ENTER_VOL_MAX < 9 and ind["vol"] * 244 ** 0.5 > ENTER_VOL_MAX:
        return False
    if DUAL_MOM and ind["mom60"] <= 0:
        return False
    if ind["pos_frac"] < POS_FRAC_MIN:
        return False
    if ind["max_ret"] > MAX_CAP:
        return False
    if MACD_FILTER and ind["dif"] <= ind["dea"]:
        return False
    if MACD_DIF_POS and ind["dif"] <= 0:
        return False
    if KDJ_GOLDEN and ind["k_val"] <= ind["d_val"]:
        return False
    if KDJ_NOCHASE and ind["j_val"] > KDJ_NC_THR:
        return False
    if RSI_NOCHASE and ind["rsi"] > RSI_NC_THR:
        return False
    if RSI_GT50 and ind["rsi"] <= 50:
        return False
    if MA_ALIGN and not ind["ma_align"]:
        return False
    if ABOVE_MA20 and not ind["above_ma20"]:
        return False
    if MA20_RISING and not ind["ma20_rising"]:
        return False
    if ind["bias20"] > BIAS_NOCHASE:
        return False
    if ind["pctb"] > PCTB_NOCHASE:
        return False
    if ind["cci"] > CCI_NOCHASE:
        return False
    if VOL_RATIO_MAX > 0 and ind["vol_ratio"] > VOL_RATIO_MAX:
        return False
    if ind["vol_in_ratio"] < VOL_IN_MIN:
        return False
    if ind["vol_in_ratio"] > VOL_IN_MAX:
        return False
    if code in GLOBAL_POOL:
        if GLOBAL_MA_FILTER and not ind["above_ma"]:
            return False
        if GLOBAL_MOM60_FILTER and ind["mom60"] <= 0:
            return False
    return True


def _panic_thr(ind):
    """panic 阈值: VOL_PANIC_ON 时按标的日波动缩放 clip(K*vol20, LO, HI), 否则固定 PANIC_DROP。"""
    if not VOL_PANIC_ON:
        return PANIC_DROP
    return min(max(VOL_PANIC_K * ind["vol"], VOL_PANIC_LO), VOL_PANIC_HI)


def _exit_hit(ind):
    """离场条件: MOM20<=EXIT_MOM_FLOOR(默认转负); 单日急跌; 过热态MOM5转负;
    动量衰减(近20日峰值回落超MOM_DECAY); 唐奇安低点(跌破前N日最低收盘)。
    MACD_EXIT/KDJ_DEAD_EXIT/RSI_LT50_EXIT 开启时叠加。"""
    if ind["mom20"] <= EXIT_MOM_FLOOR:
        return "MOM20 低于阈值 %.1f%%" % (EXIT_MOM_FLOOR * 100) if EXIT_MOM_FLOOR > 0 else "MOM20 转负"
    if PANIC_DROP > 0 and ind["ret1"] <= -_panic_thr(ind):
        return "单日急跌 %.1f%% 紧急离场" % (ind["ret1"] * 100)
    if ind["mom20_max"] > OVERHEAT and ind["mom5"] <= 0:
        return "过热回落(近%d日MOM20峰值 %+.0f%% 且 MOM5 转负)" % (
            OVERHEAT_LOOKBACK, ind["mom20_max"] * 100)
    if MOM_DECAY < 9 and ind["mom20_peak20"] - ind["mom20"] > MOM_DECAY:
        return "动量衰减(峰值 %+.1f%% -> %+.1f%%)" % (ind["mom20_peak20"] * 100, ind["mom20"] * 100)
    if DONCHIAN_N > 0 and not ind["above_donchian"]:
        return "跌破近%d日最低收盘(唐奇安)" % DONCHIAN_N
    if MACD_EXIT and ind["dif"] < ind["dea"]:
        return "MACD 死叉"
    if KDJ_DEAD_EXIT and ind["k_val"] < ind["d_val"]:
        return "KDJ 死叉"
    if RSI_LT50_EXIT and ind["rsi"] < 50:
        return "RSI 跌破 50"
    if ZSCORE_ACCEL and ind["mom10_z"] > 3.0:
        return "加速止盈(mom10 z=%.1f)" % ind["mom10_z"]
    return None


def _is_bull(info):
    """牛市体制判定。多数表决: 股票池≥2只站上年线; 否则看沪深300单基准。
    MA_SLOPE 开启时额外要求年线上行; BULL_DUAL 开启时额外要求沪深300自身MOM20>0。
    BULL_HYST>0 启用滞后带(维持原体制直至突破带外); BULL_CONFIRM>1 翻转需连续N天确认。"""
    global _state_bull, BULL_HYST_PENDING
    base = info.get(BULL_CODE, {})
    bull = base.get("above_ma", True) and base.get("ma_rising", True)
    if BULL_DUAL and base.get("mom20", 1.0) <= 0:
        bull = False
    if BULL_VOTE:
        votes = sum(1 for c in STOCK_POOL if info.get(c, {}).get("above_ma"))
        bull = votes >= 2
    raw_bull = bull
    if BULL_HYST > 0 and _state_bull is not None:
        # 滞后带: 维持原体制, 仅当明确突破带外才翻转
        dist = base.get("dist_ma250", 0.0)
        if _state_bull and dist < -BULL_HYST:
            raw_bull = False
        elif not _state_bull and dist > BULL_HYST:
            raw_bull = True
        else:
            raw_bull = _state_bull
    if BULL_CONFIRM > 1 and _state_bull is not None and raw_bull != _state_bull:
        if BULL_HYST_PENDING is not None and BULL_HYST_PENDING[0] == raw_bull:
            BULL_HYST_PENDING = (raw_bull, BULL_HYST_PENDING[1] + 1)
        else:
            BULL_HYST_PENDING = (raw_bull, 1)
        if BULL_HYST_PENDING[1] >= BULL_CONFIRM:
            _state_bull = raw_bull
            BULL_HYST_PENDING = None
        return _state_bull
    _state_bull = raw_bull
    BULL_HYST_PENDING = None
    return raw_bull


def decide(table, holding, holding_days=99):
    """返回 (target, reason)。target 为标的代码，CASH 表示空仓(停靠货币ETF)。

    竞赛池: 牛市 = A股池 + 跨境池; 熊市 = 跨境池 + 黄金(它们不受A股体制约束)。
    牛市中竞赛池无人达标时, 黄金作备胎(需 MOM20>0)。
    holding_days: 当前持仓已持有的交易日数(用于 MIN_HOLD 轮动限制)。
    """
    info = {c: ind for c, ind in table}
    bull = _is_bull(info)
    comp_pool = STOCK_POOL + GLOBAL_POOL if (bull or BEAR_OPEN_STOCK) else GLOBAL_POOL + [GOLD]
    best = next((t for t in table if t[0] in comp_pool and _enter_ok(t[1], t[0], bull)), None)

    if best:
        target, reason = best[0], "竞赛池动量第一" if bull else "熊市体制, 跨境/黄金动量第一"
    elif bull and GOLD in info and info[GOLD]["mom20"] > 0:
        target, reason = GOLD, "竞赛池无人达标, 黄金动量为正(备胎)"
    elif NEVER_EMPTY:
        # v7 永不空仓: 避险池(跨境+黄金)score最强者接盘(即便动量为负, 选跌得最少的)
        # SAFE_MIN_VOL: 保守兜底选vol最低者; NE_MIN_MOM: 兜底质量门; NE_BULL_ONLY: 仅牛市
        if NE_BULL_ONLY and not bull:
            target, reason = CASH, "熊市体制, 空仓避险"
        else:
            if SAFE_MIN_VOL:
                cands = [x for x in table if x[0] in GLOBAL_POOL + [GOLD]]
                cand = min(cands, key=lambda x: x[1]["vol"]) if cands else None
            else:
                cand = next((x for x in table if x[0] in GLOBAL_POOL + [GOLD]), None)
            if cand and cand[1]["mom20"] >= NE_MIN_MOM:
                target, reason = cand[0], "竞赛池无人达标, 避险池%s接盘(永不空仓)" % (
                    "最低波动" if SAFE_MIN_VOL else "最强")
            elif cand:
                target, reason = CASH, "避险最强 %s MOM20 %.1f%% 低于兜底门, 持币" % (
                    cand[0], cand[1]["mom20"] * 100)
            else:
                target, reason = CASH, "无人达标, 空仓避险" if bull else "熊市体制, 空仓避险"
    else:
        target, reason = CASH, "无人达标, 空仓避险" if bull else "熊市体制, 空仓避险"

    if holding and holding != target and holding in info:
        h = info[holding]
        # 领导力背离(平台移植): 宽基(510300)动量超过竞赛池最强 -> 池子失去领导力, 离场去避险
        if LEAD_EXIT and holding != CASH:
            b300 = info.get(BULL_CODE, {}).get("mom20", -9)
            best_mom = max((info[c]["mom20"] for c in comp_pool if c in info), default=-9)
            if b300 > best_mom and holding != BULL_CODE:
                cand = next((x for x in table if x[0] in GLOBAL_POOL + [GOLD] and x[0] != holding), None)
                tgt = cand[0] if cand else CASH
                return tgt, "领导力背离: 沪深300 MOM20 %+.1f%% 超池内最强 %+.1f%%, 离场" % (
                    b300 * 100, best_mom * 100)
        if not bull and not BEAR_OPEN_STOCK and holding in STOCK_POOL:
            # 制度性离场(A股池), 优先于滞回缓冲
            return target, "熊市体制确立, A股持仓无条件离场 -> %s" % UNIVERSE[target][0]
        exit_why = _exit_hit(h)
        if exit_why:
            return target, "%s, 离场 -> %s" % (exit_why, UNIVERSE[target][0])
        if MIN_HOLD and holding_days < MIN_HOLD and holding != CASH:
            return holding, "持仓满%d天前不轮动(当前%d天), 继续持有" % (MIN_HOLD, holding_days)
        # 排名缓冲(替代滞回): 持仓仍居竞赛池前N则不轮动
        if RANK_EXIT_N > 0:
            comp_ranked = [c for c, _ in table if c in comp_pool]
            if holding in comp_ranked and comp_ranked.index(holding) < RANK_EXIT_N:
                return holding, "排名缓冲: 持仓仍居竞赛池前%d, 继续持有" % RANK_EXIT_N
        # 持仓健康 -> 滞回缓冲, 挑战者须显著更强
        # v8: 震荡市(持仓ER20低)或同因子(相关系数高)时缓冲加倍; VOL_BUFFER: 按挑战者vol缩放
        buf = BUFFER
        if POOL_BUFFER and target in info:  # 分池缓冲: 挑战者所在池决定
            _role = UNIVERSE.get(target, ("", "", ""))[2]
            buf = POOL_BUFFER.get("gold" if target == GOLD else _role, BUFFER)
        if TREND_BUF_ON and h["mom20"] > 0.10:  # 趋势强度自适应: 强趋势持仓缓冲升至5%
            buf = max(buf, 0.05)
        if SLOPE_DAYS_BUF > 0 and h.get("slope_pos_days", 0) >= SLOPE_DAYS_BUF:
            buf *= 2
        if VOL_BUFFER > 0 and target in info:
            buf = max(BUFFER, VOL_BUFFER * info[target]["vol"] * 20 ** 0.5)
        if ER_BUFFER_ON and h.get("er20", 1.0) < KAUFMAN_THR:
            buf *= 2
        if REL_BUFFER_ON and target in info:
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
        c_mom = info[target]["mom20"] if target in info else 0.0
        if c_mom - h["mom20"] < buf:
            return holding, "滞回缓冲: 挑战者(%s %+.2f%%) 未超持仓(%s %+.2f%%) %.0f%%, 继续持有" % (
                UNIVERSE[target][0], c_mom * 100,
                UNIVERSE[holding][0], h["mom20"] * 100, buf * 100)
        reason += " (挑战者显著更强, 轮动)"
    return target, reason


def advice(table, holding):
    """生成 (target, reason, 操作文本, 排名表文本)。"""
    target, reason = decide(table, holding)
    bull = _is_bull({c: ind for c, ind in table})
    lines = ["  牛熊体制: %s (%s)" % ("牛市" if bull else "熊市",
             "股票ETF多数表决" if BULL_VOTE else "沪深300ETF 相对年线")]
    for code, ind in table:
        tag = " ← 当前持仓" if code == holding else ""
        lines.append("  %-7s %s  MOM20 %+6.2f%%  MOM60 %+6.2f%%  score %+6.2f%s"
                     % (code, UNIVERSE[code][0], ind["mom20"] * 100,
                        ind["mom60"] * 100, ind["score"], tag))
    if holding == target:
        act = "继续持有 %s %s" % (target, UNIVERSE[target][0]) if target != CASH else "继续空仓(可停靠511880)"
    elif holding:
        act = "卖出 %s %s -> 买入 %s %s" % (holding, UNIVERSE[holding][0], target, UNIVERSE[target][0]) \
            if target != CASH else "卖出 %s %s -> 空仓(或买511880)" % (holding, UNIVERSE[holding][0])
    else:
        act = "买入 %s %s" % (target, UNIVERSE[target][0]) if target != CASH else "保持空仓"
    return target, reason, act, "\n".join(lines)
