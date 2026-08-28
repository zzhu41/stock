# -*- coding: utf-8 -*-
"""历史回测: 模拟实盘尾盘操作 —— T日收盘价出信号并以收盘价成交, T+1起按新持仓计收益。
空仓时持有511880货币ETF(真实数据)。费用: 单边万一, 换仓双边万二。
"""
import sys
from market_data import UNIVERSE, STOCK_POOL, GLOBAL_POOL, GOLD, CASH, fetch_history
import strategy

FEE = 0.0001          # 单边佣金万一
TRADING_DAYS = 244    # A股年均交易日
START = "2018-01-01"


def backtest(histories, calendar, buffer=0.02, overheat=0.40, bull_vote=False,
             circuit_dd=0.0, circuit_days=10, skip_days=0, use_high52=False,
             min_hold=0, start=START, end="9999", overheat_lookback=1,
             panic_drop=0.04, global_ma_filter=False, global_mom60_filter=False,
             pos_frac_min=0.0, max_cap=9.9, use_downside_vol=False, vol_target=0.0,
             vol_target_lock=False, mom_main=20, ma_bull=250,
             dual_mom=False, score_mom=0, ma_slope=False, rank_exit_n=0,
             score_plain_mom=False, leverage=1.0, borrow_rate=0.06, never_empty=True,
             exit_cooldown=0, abs_stop=0.0, macd_filter=False, macd_exit=False,
             kdj_nochase=False, kdj_dip_buy=False, rsi_nochase=False, ma_align=False,
             macd_dif_pos=False, kdj_golden=False, kdj_dead_exit=False,
             rsi_gt50=False, rsi_lt50_exit=False, above_ma20=False, ma20_rising=False,
             bias_nochase=9.9, pctb_nochase=9.9, cci_nochase=9999.0,
             kdj_nc_thr=100.0, rsi_nc_thr=80.0, score_tech_mix=False,
             ne_min_mom=-9.9, ne_bull_only=False, exit_mom_floor=0.0,
             mom_decay=9.9, donchian_n=0,
             er_buffer_on=False, kaufman_thr=0.30, rel_buffer_on=False,
             vol_panic_on=False, signal_histories=None, w_mom=False, vol_ratio_max=0.0,
             enter_mom_min=0.0, score_square=False, score_wls=True,
             trail_stop=0.0, trail_cool=0,
             vol_in_min=0.0, vol_in_max=99.0, vol_score_mix=False,
             score_smooth=1, enter_vol_max=9.9, bear_enter_mom=0.07,
             safe_min_vol=False, vol_buffer=0.0, high_vol_half=0.0, dd_guard=0.0,
             score_r2=False, wls60_mix=0.0, vol_ewm=False, bull_dual=False,
             vol_days=20, bear_open_stock=False, crash_mom5=-0.08, crash_lock=5,
             crash_stock_only=False, crash_below_ma=0.20,
             crash_stop=0.0, crash_pick="score", crash_alloc=1.0, score_tstat=False,
             lead_exit=False, zscore_crash=False, zscore_accel=False,
             bull_hyst=0.0, bull_confirm=1, wls_adaptive=False, resid_penalty=False,
             crash_dyn_unlock=False, crash_tp=0.0,
             trend_buf_on=False, pool_buffer=None, slope_days_buf=0):
    """v9.1 默认: v9 + 分池缓冲(pool_buffer=None 时引擎内取 strategy.POOL_BUFFER)。"""
    """v9 默认: v8.1 + 深跌恐慌抄底(crash_mom5=-8%%, 低于年线20%%, 锁仓5天)。"""
    strategy.BUFFER = buffer
    strategy.OVERHEAT = overheat
    strategy.BULL_VOTE = bull_vote
    strategy.SKIP_DAYS = skip_days
    strategy.USE_HIGH52 = use_high52
    strategy.MIN_HOLD = min_hold
    strategy.OVERHEAT_LOOKBACK = overheat_lookback
    strategy.PANIC_DROP = panic_drop
    strategy.GLOBAL_MA_FILTER = global_ma_filter
    strategy.GLOBAL_MOM60_FILTER = global_mom60_filter
    strategy.POS_FRAC_MIN = pos_frac_min
    strategy.MAX_CAP = max_cap
    strategy.USE_DOWNSIDE_VOL = use_downside_vol
    strategy.DUAL_MOM = dual_mom
    strategy.SCORE_MOM = score_mom
    strategy.MA_SLOPE = ma_slope
    strategy.RANK_EXIT_N = rank_exit_n
    strategy.SCORE_PLAIN_MOM = score_plain_mom
    strategy.NEVER_EMPTY = never_empty
    strategy.MACD_FILTER = macd_filter
    strategy.MACD_EXIT = macd_exit
    strategy.KDJ_NOCHASE = kdj_nochase
    strategy.KDJ_DIP_BUY = kdj_dip_buy
    strategy.RSI_NOCHASE = rsi_nochase
    strategy.MA_ALIGN = ma_align
    strategy.MACD_DIF_POS = macd_dif_pos
    strategy.KDJ_GOLDEN = kdj_golden
    strategy.KDJ_DEAD_EXIT = kdj_dead_exit
    strategy.RSI_GT50 = rsi_gt50
    strategy.RSI_LT50_EXIT = rsi_lt50_exit
    strategy.ABOVE_MA20 = above_ma20
    strategy.MA20_RISING = ma20_rising
    strategy.BIAS_NOCHASE = bias_nochase
    strategy.PCTB_NOCHASE = pctb_nochase
    strategy.CCI_NOCHASE = cci_nochase
    strategy.KDJ_NC_THR = kdj_nc_thr
    strategy.RSI_NC_THR = rsi_nc_thr
    strategy.SCORE_TECH_MIX = score_tech_mix
    strategy.NE_MIN_MOM = ne_min_mom
    strategy.NE_BULL_ONLY = ne_bull_only
    strategy.EXIT_MOM_FLOOR = exit_mom_floor
    strategy.MOM_DECAY = mom_decay
    strategy.DONCHIAN_N = donchian_n
    strategy.ER_BUFFER_ON = er_buffer_on
    strategy.KAUFMAN_THR = kaufman_thr
    strategy.REL_BUFFER_ON = rel_buffer_on
    strategy.VOL_PANIC_ON = vol_panic_on
    strategy.W_MOM = w_mom
    strategy.VOL_RATIO_MAX = vol_ratio_max
    strategy.ENTER_MOM_MIN = enter_mom_min
    strategy.SCORE_SQUARE = score_square
    strategy.SCORE_WLS = score_wls
    strategy.VOL_IN_MIN = vol_in_min
    strategy.VOL_IN_MAX = vol_in_max
    strategy.VOL_SCORE_MIX = vol_score_mix
    strategy.SCORE_SMOOTH = score_smooth
    strategy.ENTER_VOL_MAX = enter_vol_max
    strategy.BEAR_ENTER_MOM = bear_enter_mom
    strategy.SAFE_MIN_VOL = safe_min_vol
    strategy.VOL_BUFFER = vol_buffer
    strategy.SCORE_R2 = score_r2
    strategy.WLS60_MIX = wls60_mix
    strategy.VOL_EWM = vol_ewm
    strategy.BULL_DUAL = bull_dual
    strategy.VOL_DAYS = vol_days
    strategy.BEAR_OPEN_STOCK = bear_open_stock
    strategy.SCORE_TSTAT = score_tstat
    strategy.LEAD_EXIT = lead_exit
    strategy.ZSCORE_CRASH = zscore_crash
    strategy.ZSCORE_ACCEL = zscore_accel
    strategy.BULL_HYST = bull_hyst
    strategy.BULL_CONFIRM = bull_confirm
    strategy._state_bull = None
    strategy.BULL_HYST_PENDING = None
    strategy.WLS_ADAPTIVE = wls_adaptive
    strategy.RESID_PENALTY = resid_penalty
    strategy.TREND_BUF_ON = trend_buf_on
    strategy.POOL_BUFFER = pool_buffer if pool_buffer is not None else \
        {"stock": 0.02, "global": 0.03, "gold": 0.03}  # v9.1 默认分池缓冲
    strategy.SLOPE_DAYS_BUF = slope_days_buf
    sig_histories = signal_histories if signal_histories is not None else histories
    strategy.MOM_MAIN = mom_main
    strategy.MA_BULL = ma_bull
    strategy.MIN_ROWS = max(strategy.MA_BULL, max(strategy.MOM_WINDOWS) + 1 + skip_days,
                            mom_main + 1 + skip_days) + strategy.VOL_DAYS
    days = [d for d in calendar if start <= d <= end]
    close_of = {c: {r[0]: r[2] for r in rows} for c, rows in histories.items()}
    cash_close = close_of[CASH]

    nav, peak, max_dd = 1.0, 1.0, 0.0
    holding, switches, daily, trades = None, 0, [], []
    holding_days = 0
    pos = 1.0              # 昨日信号定下的风险资产仓位(其余停靠货币ETF); vol_target=0 时恒为1
    circuit_until = -1  # 熔断冷却截止下标
    cooldown_until = -1  # 风控离场冷却截止下标(参考平台空仓冷却): 期内不得买竞赛池
    cost_price = None   # 当前持仓成本价(绝对止损用)
    hold_peak = None    # 持仓期最高价(trailing止盈用)
    guard_on = False    # 组合回撤风控状态(dd_guard): True=仅允许避险池
    lock_until = -1     # 抄底锁仓截止下标(crash_lock): 期内强制持有不操作
    crash_buys = []     # 抄底记录(分析用)
    for i, d in enumerate(days):
        if i > 0:  # 当日收益按"昨日定下的持仓与仓位"计算
            prev, cur = close_of[holding].get(days[i - 1]), close_of[holding].get(d)
            if prev and cur:
                r_asset = cur / prev - 1.0
                if holding != CASH and leverage > 1.0:
                    # 杠杆: 收益放大, 超出本金部分付融资利息(空仓持币期不上杠杆)
                    nav *= 1.0 + leverage * r_asset - (leverage - 1.0) * borrow_rate / TRADING_DAYS
                elif pos < 1.0:
                    c0, c1 = cash_close.get(days[i - 1]), cash_close.get(d)
                    r_cash = (c1 / c0 - 1.0) if (c0 and c1) else 0.0
                    nav *= 1.0 + pos * r_asset + (1.0 - pos) * r_cash
                else:
                    nav *= cur / prev
            holding_days += 1
        if circuit_dd > 0 and nav / peak - 1 <= -circuit_dd and i > circuit_until:
            circuit_until = i + circuit_days   # 触发熔断
        table = strategy.rank(sig_histories, on_date=d)
        target, _ = strategy.decide(table, holding, holding_days)
        if i < circuit_until:
            target = CASH                       # 熔断冷却期强制空仓
        info = {c: ind for c, ind in table}
        # 组合回撤风控(dd_guard): 净值回撤超阈值期间, 新目标仅允许避险池(修复一半后解除)
        if dd_guard > 0:
            cur_dd = nav / peak - 1.0
            if not guard_on and cur_dd <= -dd_guard:
                guard_on = True
            elif guard_on and cur_dd >= -dd_guard / 2.0:
                guard_on = False
            if guard_on and target in STOCK_POOL + GLOBAL_POOL:
                cand = next((x for x in table if x[0] in strategy.GLOBAL_POOL + [GOLD]
                             and x[0] != holding), None)
                target = cand[0] if cand else CASH
        # 恐慌抄底(crash_mom5<0开启): 指定范围标的MOM5<=-阈值 -> 主动买入, 锁仓crash_lock天
        if i < lock_until:
            target = holding                    # 锁仓期不操作
            # 锁仓期止损补丁(crash_stop>0): 抄底成本再跌超X%提前解锁去避险
            if crash_stop > 0 and holding and holding != CASH and cost_price:
                cur = close_of[holding].get(d)
                if cur and cur / cost_price - 1.0 <= -crash_stop:
                    cand = next((x for x in table if x[0] in strategy.GLOBAL_POOL + [GOLD]
                                 and x[0] != holding), None)
                    target = cand[0] if cand else CASH
                    lock_until = i              # 提前解锁
            # 大涨止盈(crash_tp>0): 锁仓期单日涨幅>=X%提前落袋(回正常信号)
            if crash_tp > 0 and holding and holding != CASH and holding in info:
                if info[holding]["ret1"] >= crash_tp:
                    target, _ = strategy.decide(table, holding, holding_days)
                    lock_until = i
            # 动态解锁(crash_dyn_unlock): MOM5转正且浮盈>3%提前解锁
            if crash_dyn_unlock and holding and holding != CASH and holding in info and cost_price:
                cur = close_of[holding].get(d, cost_price)
                if info[holding]["mom5"] > 0 and cur / cost_price - 1.0 > 0.03:
                    target, _ = strategy.decide(table, holding, holding_days)
                    lock_until = i
        elif crash_mom5 < 0:
            crash_pool = STOCK_POOL if crash_stock_only else (STOCK_POOL + GLOBAL_POOL + [GOLD])
            if zscore_crash:  # 平台版触发: mom10 z<-2.5 且价格低于年线
                cands = [x for x in table if x[0] in crash_pool and x[0] != holding
                         and x[1]["mom10_z"] <= -2.5 and x[1]["dist_ma250"] < 0]
            else:
                cands = [x for x in table if x[0] in crash_pool
                         and x[1]["mom5"] <= crash_mom5 and x[0] != holding
                         and (crash_below_ma <= 0
                              or x[1]["dist_ma250"] < -crash_below_ma)]
            if crash_pick == "lowest":          # 选跌最狠的(纯反转)
                cand = min(cands, key=lambda x: x[1]["mom5"]) if cands else None
            else:                               # 默认: score最强的深跌者
                cand = cands[0] if cands else None
            if cand:
                target = cand[0]
                lock_until = i + crash_lock
                crash_buys.append((d, cand[0]))
        # 绝对止损(参考平台): 持仓浮亏超 abs_stop 强制离场去避险兜底
        risk_exit = False
        if abs_stop > 0 and holding and holding != CASH and cost_price and target == holding:
            cur = close_of[holding].get(d)
            if cur and cur / cost_price - 1.0 <= -abs_stop:
                cand = next((x for x in table if x[0] in strategy.GLOBAL_POOL + [GOLD]), None)
                target = cand[0] if cand else CASH
                risk_exit = True
        # trailing 止盈(次方量化同款): 持仓期最高价回撤超 trail_stop 强制止盈+冷却
        if trail_stop > 0 and holding and holding != CASH:
            cur = close_of[holding].get(d)
            if cur:
                hold_peak = max(hold_peak or cur, cur)
                if cur < hold_peak * (1.0 - trail_stop) and target == holding:
                    cand = next((x for x in table if x[0] in strategy.GLOBAL_POOL + [GOLD]), None)
                    target = cand[0] if cand else CASH
                    risk_exit = True
                    cooldown_until = i + trail_cool
        # 识别风控离场(decide因exit_hit/熊市离场), 触发冷却
        if (exit_cooldown > 0 and holding and holding != CASH and target != holding
                and holding in info):
            bull_now = strategy._is_bull(info)
            if strategy._exit_hit(info[holding]) or (not bull_now and holding in STOCK_POOL):
                risk_exit = True
        if risk_exit:
            cooldown_until = i + exit_cooldown
        # 冷却期内不得买竞赛池标的 -> 已持健康避险则不动, 否则避险兜底
        if i < cooldown_until and target in STOCK_POOL + GLOBAL_POOL:
            if (holding in strategy.GLOBAL_POOL + [GOLD] and holding in info
                    and not strategy._exit_hit(info[holding])):
                target = holding
            else:
                cand = next((x for x in table if x[0] in strategy.GLOBAL_POOL + [GOLD]
                             and x[0] != holding), None)
                target = cand[0] if cand else CASH
        if target != holding:
            if i > 0:                # 首日建仓不计为换手
                switches += 1
                nav *= (1 - FEE * 2) # 双边费用
                trades.append((d, holding, target, nav))
            holding = target
            holding_days = 0
            cost_price = close_of[holding].get(d) if holding != CASH else None
            hold_peak = cost_price
            if vol_target_lock:      # 简化模式: 仅换仓日按当时vol定仓位并锁定(实盘友好)
                v = info.get(holding, {}).get("vol", 0.0)
                pos = min(1.0, vol_target / (v * TRADING_DAYS ** 0.5)) if (v > 0 and holding != CASH) else 1.0
        # 波动率目标(Moreira & Muir 2017): 仓位=目标年化波动/标的年化波动, 上限满仓; 调仓不收费(近似)
        if vol_target > 0 and not vol_target_lock and holding != CASH:
            info = {c: ind for c, ind in table}
            v = info.get(holding, {}).get("vol", 0.0)
            pos = min(1.0, vol_target / (v * TRADING_DAYS ** 0.5)) if v > 0 else 1.0
        elif not vol_target_lock:
            pos = 1.0
        # 极端高波临时半仓(high_vol_half): 持仓年化vol>阈值时半仓(余仓货币), 调仓不收费
        if high_vol_half > 0 and holding != CASH:
            v = info.get(holding, {}).get("vol", 0.0)
            if v * TRADING_DAYS ** 0.5 > high_vol_half:
                pos = 0.5
        # 抄底仓位(crash_alloc<1): 锁仓期内部分仓位(余仓货币)
        if crash_alloc < 1.0 and i < lock_until and holding != CASH:
            pos = crash_alloc
        peak = max(peak, nav)
        max_dd = min(max_dd, nav / peak - 1)
        daily.append((d, nav, holding))

    n = len(daily)
    rets = [daily[i][1] / daily[i - 1][1] - 1 for i in range(1, n)]
    mean = sum(rets) / len(rets)
    std = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
    years = n / TRADING_DAYS
    ann = nav ** (1 / years) - 1
    return {
        "nav": nav, "ann": ann, "max_dd": max_dd,
        "sharpe": mean / std * TRADING_DAYS ** 0.5 if std else 0,
        "calmar": ann / abs(max_dd) if max_dd else 0,
        "switches": switches, "sw_per_year": switches / years,
        "daily": daily, "trades": trades, "crash_buys": crash_buys,
    }


def buy_and_hold(histories, calendar, code, start=START, end="9999"):
    close_of = {r[0]: r[2] for r in histories[code]}
    days = [d for d in calendar if start <= d <= end and d in close_of]
    nav, peak, max_dd = 1.0, 1.0, 0.0
    for i in range(1, len(days)):
        nav *= close_of[days[i]] / close_of[days[i - 1]]
        peak = max(peak, nav)
        max_dd = min(max_dd, nav / peak - 1)
    years = len(days) / TRADING_DAYS
    return nav, nav ** (1 / years) - 1, max_dd


def report(res, title):
    print("\n== %s ==" % title)
    print("总收益 %+7.1f%% | 年化 %+5.1f%% | 最大回撤 %5.1f%% | 夏普 %.2f | 卡玛 %.2f | 换手 %d次(年均%.1f)"
          % ((res["nav"] - 1) * 100, res["ann"] * 100, res["max_dd"] * 100,
             res["sharpe"], res["calmar"], res["switches"], res["sw_per_year"]))


def yearly(daily):
    """逐年收益: 每年最后净值 / 上年最后净值 - 1。"""
    last = {}
    for d, nav, _ in daily:
        last[d[:4]] = nav
    res, base = [], 1.0
    for y in sorted(last):
        res.append((y, last[y] / base - 1))
        base = last[y]
    return res


if __name__ == "__main__":
    codes = list(UNIVERSE)
    histories = {c: fetch_history(c) for c in codes}
    calendar = [r[0] for r in histories["510300"]]
    start = sys.argv[1] if len(sys.argv) > 1 else START
    print("回测区间 %s ~ %s | v9 默认全组件 | 费用双边万二" % (start, calendar[-1]))

    res = backtest(histories, calendar, start=start)
    report(res, "动量轮动策略 v9")
    for y, r in yearly(res["daily"]):
        print("  %s: %+7.1f%%" % (y, r * 100))

    for c in ["510300", "518880"]:
        nav, ann, dd = buy_and_hold(histories, calendar, c, start=start)
        print("基准 %s %s 买入持有: 总收益 %+6.1f%% 年化 %+5.1f%% 回撤 %5.1f%%"
              % (c, UNIVERSE[c][0], (nav - 1) * 100, ann * 100, dd * 100))

    print("\n== 组件消融: 关闭后年化/回撤/夏普/年换手 (与基线差即该组件贡献) ==")
    variants = [
        ("基线 v9          ", {}),
        ("去急跌离场panic  ", {"panic_drop": 0.0}),
        ("去永不空仓       ", {"never_empty": False}),
        ("去WLS25(回v7排名)", {"score_wls": False}),
        ("去熊市门槛7%%     ", {"bear_enter_mom": 0.0}),
        ("去深跌抄底       ", {"crash_mom5": 0.0}),
        ("去滞回缓冲2%%     ", {"buffer": 0.0}),
        ("去过热快离场     ", {"overheat": 9.9}),
    ]
    for name, kw in variants:
        r = backtest(histories, calendar, start=start, **kw)
        yrs = " ".join("%s:%+.0f" % (y[2:], x * 100) for y, x in yearly(r["daily"]))
        print("%s: %+5.1f%% | %6.1f%% | %4.2f | %4.1f || %s"
              % (name, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"],
                 r["sw_per_year"], yrs))

    print("\n== 最近一期信号(基于最新收盘) ==")
    backtest(histories, calendar, start="2020-01-01")  # 重置 strategy 全局参数为默认
    table = strategy.rank(histories)
    holding = res["daily"][-1][2]
    target, reason, act, tbl = strategy.advice(table, holding)
    print(tbl)
    print("回测持仓: %s %s | 建议: %s (%s)" % (holding, UNIVERSE[holding][0], act, reason))
