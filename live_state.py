# -*- coding: utf-8 -*-
"""实时信号共用的交易日锁仓和影子估值，不读取或写入账户文件。"""
import math

from market_data import UNIVERSE

VALUATION_VERSION = 3


def signal_calendar(histories, signal_date):
    """只接受已由行情层对齐到实际信号日的行情。"""
    cal = [r[0] for r in histories.get("510300", [])]
    if not cal or cal[-1] != signal_date:
        raise ValueError("信号日 %s 与沪深300行情末日不一致" % signal_date)
    for code, rows in histories.items():
        if not rows or rows[-1][0] != signal_date:
            raise ValueError("%s 缺少信号日 %s 的行情，停止影子记账" % (code, signal_date))
    return cal


def lock_active(code, trigger_date, calendar, signal_date, days=5):
    """触发日为第0日，第5个后续交易日恢复决策，与回测 i < i0+5 一致。"""
    if not code:
        return False
    if code not in UNIVERSE:
        raise ValueError("锁仓标的无效: %s" % code)
    if not trigger_date:
        raise ValueError("旧锁仓缺少 trigger_date，无法可靠迁移；请核对原始触发记录")
    if trigger_date not in calendar or signal_date not in calendar or trigger_date > signal_date:
        raise ValueError("锁仓触发日 %s 无法在信号日历中确认" % trigger_date)
    return calendar.index(signal_date) - calendar.index(trigger_date) < days


def migrate_shadow_state(st, signal_date):
    """旧净值不具备信号价口径：归档原值，并显式开始一条新的前向净值。"""
    if st.get("last_date") and st["last_date"] > signal_date:
        raise ValueError("信号日期早于已记账日期，停止影子记账")
    if st.get("valuation_version") == VALUATION_VERSION:
        return
    if st.get("lock_code"):
        trigger = st.get("lock_trigger_date") or st.get("trigger_date")
        if not trigger:
            raise ValueError("旧影子锁仓没有触发日，无法可靠迁移；旧状态保留")
        st["lock_trigger_date"] = trigger
    if st.get("last_date"):
        if st.get("legacy_performance"):
            st.setdefault("legacy_performance_history", []).append(st["legacy_performance"])
        st["legacy_performance"] = {
            "nav": st.get("nav", 1.0), "start_date": st.get("start_date"),
            "last_date": st["last_date"], "valuation_version": st.get("valuation_version", 1),
            "valuation": ("signal_price_without_adjustment_anchor"
                          if st.get("valuation_version") == 2 else "historical_close_legacy"),
        }
        st["valuation_note"] = (
            "⚠️ 旧口径净值 %.6f（截至%s）已归档，不作前向业绩；"
            "%s 起按信号时价格重新从1记账，并校正分红拆分"
            % (st.get("nav", 1.0), st["last_date"], signal_date))
    st.update(valuation_version=VALUATION_VERSION, nav=1.0, start_date=signal_date,
              last_date=None, mark_price=None, mark_code=None,
              mark_anchor_date=None, mark_anchor_close=None)
    st.pop("lock_until", None)


def mark_price(prices, code):
    if code is None:
        return None
    p = prices.get(code)
    if p is None or not math.isfinite(p) or p <= 0:
        raise ValueError("%s 缺少有效当日估值价格，停止影子记账" % code)
    return p


def _anchor_close(histories, code, anchor_date):
    matches = [r[2] for r in histories.get(code, []) if r[0] == anchor_date]
    if len(matches) != 1 or not math.isfinite(matches[0]) or matches[0] <= 0:
        raise ValueError("%s 缺少有效的已完成日复权锚 %s，停止影子记账" % (code, anchor_date))
    return matches[0]


def _completed_anchor(histories, code, signal_date):
    if code is None:
        return None, None
    completed = [r[0] for r in histories.get(code, []) if r[0] < signal_date]
    if not completed:
        raise ValueError("%s 没有信号日前已完成行情，无法建立复权锚" % code)
    anchor_date = max(completed)
    return anchor_date, _anchor_close(histories, code, anchor_date)


def advance_shadow_nav(st, prices, histories):
    """以已完成日的前复权重定基比例校正执行价，计算含分红拆分的持仓收益。

    锚必须早于上次信号日，不能把该日盘中价到收盘价的变化误当成复权变化。
    与行情回测一致采用前复权收益口径，不单独模拟现金分红到账。
    """
    holding = st.get("holding")
    if not st.get("last_date") or not holding:
        return
    previous = st.get("mark_price")
    if st.get("mark_code") != holding or previous is None or not math.isfinite(previous) or previous <= 0:
        raise ValueError("影子持仓与上次估值价格不一致，停止记账并保留原状态")
    anchor_date, anchor_close = st.get("mark_anchor_date"), st.get("mark_anchor_close")
    if (not anchor_date or anchor_date >= st["last_date"] or anchor_close is None
            or not math.isfinite(anchor_close) or anchor_close <= 0):
        raise ValueError("影子状态缺少可靠的已完成日复权锚，停止记账并保留原状态")
    previous *= _anchor_close(histories, holding, anchor_date) / anchor_close
    st["nav"] *= mark_price(prices, holding) / previous


def settle_shadow(st, target, prices, signal_date, fee, log_trade, histories):
    """按当前信号价建立下一段估值基准；首次建基准不回算任何入场前收益。"""
    price = mark_price(prices, target)
    anchor_date, anchor_close = _completed_anchor(histories, target, signal_date)
    if target != st.get("holding"):
        if st.get("last_date"):
            st["nav"] *= 1 - fee
            log_trade(signal_date, st.get("holding"), target, price or 0.0, st["nav"])
        st["holding"] = target
    st.update(last_date=signal_date, mark_price=price, mark_code=target,
              mark_anchor_date=anchor_date, mark_anchor_close=anchor_close)
