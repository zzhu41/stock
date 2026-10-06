# -*- coding: utf-8 -*-
"""14:30 preview push: a live estimate 20 minutes before the authoritative bundle.

The 14:50–14:55 pipeline is unchanged and stays the only commit point. This
best-effort preview reuses momentum_live.live_payload (observe_only staging:
production accounts, the canonical journal and every authoritative status file
are never touched), labels its quote time and pushes one markdown card. It
never writes generation_status.json / daily_state.json / delivery.json — the
watchdog and the sentinel must keep judging only the 14:50 chain; a preview
success can never mask a real-chain failure. Skips are silent: non-trading
day, outside the 14:30–14:40 window, a previous preview still running, or the
saved bundle already covering today's quotes (late run after the real commit).
"""
import os
from pathlib import Path
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import signal_store
import live_utils
import momentum_live

WINDOW_START, WINDOW_END = "14:30:00", "14:40:00"


def in_window(now):
    return WINDOW_START <= now.strftime("%H:%M:%S") < WINDOW_END


def main(now=None, send=None):
    now = now or signal_store.now_local()
    date = now.strftime("%Y-%m-%d")
    directory = Path(signal_store.SIGNALS)
    if signal_store.trading_day(date) is False:
        print("预推送跳过: 交易所休市日")
        return 0
    if not in_window(now):
        print("预推送跳过: 不在14:30–14:40窗口内")
        return 0
    try:
        with signal_store.file_lock(directory / "preview.lock"):
            payload = momentum_live.live_payload(momentum_live._load_journal(), now)
            if payload is None:
                # 当日正式信号已提交(14:50后手动运行时), 保存记录即权威, 不再预告。
                print("预推送跳过: 当日正式信号已覆盖最新行情")
                return 0
            markdown = ("🔔 动量预推送（14:30 数据，正式信号以 14:50 为准）\n\n"
                        + payload["markdown"])
            send = send or live_utils.send_dingtalk
            sent = send(markdown, title="动量预推送")
            signal_store.atomic_json(directory / "preview_status.json",
                                     dict(date=date, status="sent" if sent else "failed",
                                          quote_time=payload.get("quote_time"),
                                          finished_at=now.isoformat()))
            print("预推送%s: 行情 %s" % ("已发送" if sent else "发送失败(仅记录, 不影响14:50正式推送)",
                                       payload.get("quote_time")))
            return 0 if sent else 1
    except BlockingIOError:
        print("预推送跳过: 上一次任务仍在执行")
        return 0
    except Exception as exc:
        print("预推送失败: %s(仅记录, 不影响14:50正式推送)" % type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
