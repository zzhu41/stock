# -*- coding: utf-8 -*-
"""实时信号共用的交易日锁仓和影子估值，不读取或写入账户文件。"""
from market_data import UNIVERSE

VALUATION_VERSION = 4


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


# Monetary accounting lives in shadow_account.py. Deliberately do not retain
# the old QFQ-ratio valuation helpers: additive cash adjustments are not share
# multipliers and must never be used as a production fallback.
