"""Bounded optional enrichment worker. No message sending; shared dated inputs."""
from concurrent.futures import ThreadPoolExecutor
import json
import sys


def collect(payload):
    from v10_live.data import build_live_view
    import premium
    import shadow_0906
    import shadow_v92
    from v10_live import runtime
    quotes, date = payload["quotes"], payload["date"]
    pool = ThreadPoolExecutor(max_workers=3)
    futures = {
        "view": pool.submit(build_live_view, quotes, date),
        "qvix": pool.submit(shadow_0906.qvix_state, date),
        "premium": pool.submit(premium.signal_block, quotes=quotes),
    }
    output = []
    try:
        output.extend(futures["premium"].result(timeout=18))
    except Exception:
        output.append("  QDII 溢价: 本次获取失败，请另行核验")
    try:
        qvix = futures["qvix"].result(timeout=22)
    except Exception:
        qvix = (None, None, None, False, "⚠️ 当日QVIX不可用，恐慌通道停用")
    try:
        view = futures["view"].result(timeout=45)
        error = None
    except Exception as exc:
        view, error = None, "原始价格/分红数据核验失败(%s)" % type(exc).__name__
    finally:
        pool.shutdown(wait=False)
    for name, module in (("v9.1-0906", shadow_0906), ("v9.2", shadow_v92)):
        try:
            output.extend(module.block(payload["table"], payload["histories"],
                {c: q["price"] for c, q in quotes.items()}, signal_date=date,
                quotes=quotes, action_view=view, qvix=qvix))
        except Exception as exc:
            output.extend(["-" * 56, "⚠️ 影子 %s 计算失败: %s" % (name, str(exc)[:180])])
    try:
        def checked_view(*args, **kwargs):
            if view is None:
                raise ValueError(error)
            return view
        output.extend(runtime.run(quotes, date, build_view=checked_view))
    except Exception as exc:
        output.extend(["-" * 56, "【影子 V10-H】虚拟跟踪不下单",
                       "  信号状态: 本次无有效建议，计算失败",
                       "  数据说明: " + str(exc)[:180]])
    return output


if __name__ == "__main__":
    try:
        print(json.dumps({"lines": collect(json.load(sys.stdin))}, ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__}))
        raise SystemExit(1)
