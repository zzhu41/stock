"""Read-only audit of committed V12-R2 forward marks through 2026-11-16.

Run: MPLCONFIGDIR=/tmp/v12_audit_mpl python3.8 -B research/v12_postrelease_20261116/audit.py
Only writes audit.json, daily.csv, and forward_nav.png beside this script.
Returns, drawdowns, and segments are signal-time shadow marks, not broker fills.
"""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
START, END = "2026-09-29", "2026-11-16"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    sources = {}

    def read(path):
        raw = (ROOT / path).read_bytes()
        sources[path] = hashlib.sha256(raw).hexdigest()
        return raw.decode("utf-8")

    canonical = json.loads(read("signals/daily_state.json"))
    assert canonical["signal_id"] == digest(canonical["text"])
    assert canonical["checksum"] == digest(json.dumps(
        {k: v for k, v in canonical.items() if k != "checksum"},
        sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    account = canonical["account_states"]["shadow_v12_r2.json"]
    events = [x for x in account["events"] if START <= x["date"] <= END]
    assert events[0]["date"] == START and events[-1]["date"] == END
    assert len({x["date"] for x in events}) == len(events)
    assert events[0]["nav"] == 1.0
    projection_checks, strategy_comparison = {}, {}
    for name, state in canonical["account_states"].items():
        projection_checks[name] = state == json.loads(read("signals/" + name))
        comparison = [x for x in state["events"] if START <= x["date"] <= END]
        cols = ("date", "to", "nav")
        strategy_comparison[name] = {
            "observations": len(comparison),
            "same_dates_targets_nav": [tuple(x[k] for k in cols) for x in events]
            == [tuple(x[k] for k in cols) for x in comparison],
            "return": comparison[-1]["nav"] / comparison[0]["nav"] - 1,
        }
    calendar = json.loads(read("calendars/sse.json"))["2026"]["closed_ranges"]
    day = dt.date.fromisoformat(START)
    expected = []
    while day.isoformat() <= END:
        date = day.isoformat()
        if day.weekday() < 5 and not any(a <= date <= b for a, b in calendar):
            expected.append(date)
        day += dt.timedelta(days=1)
    actual = [x["date"] for x in events]
    assert expected == actual
    rows, peak, mdd, mdd_dates = [], events[0], 0.0, None
    previous_nav = 1.0
    reasons = {}
    for event in events:
        text = read("signals/" + event["date"] + ".txt")
        assert "生成 " + event["date"] in text
        line = next(x for x in text.splitlines() if "影子持仓:" in x)
        assert event["to"] in line and format(event["nav"], ".4f") in line
        reasons[event["date"]] = line.rsplit(" | ", 1)[-1]
        if event["nav"] > peak["nav"]:
            peak = event
        drawdown = event["nav"] / peak["nav"] - 1
        if drawdown < mdd:
            mdd, mdd_dates = drawdown, [peak["date"], event["date"]]
        rows.append(dict(date=event["date"], nav=event["nav"],
                         daily_return=event["nav"] / previous_nav - 1,
                         drawdown=drawdown, return_holding=event["from"],
                         closing_holding=event["to"], fee_fraction=event["fee_fraction"],
                         mark_raw_price=event["mark_raw_price"], reason=reasons[event["date"]]))
        previous_nav = event["nav"]
    starts = [i for i, x in enumerate(events) if x["from"] != x["to"]]
    segments = []
    for pos, begin in enumerate(starts):
        closed = pos + 1 < len(starts)
        end = starts[pos + 1] if closed else len(events) - 1
        first, last = events[begin], events[end]
        gross_exit_nav = last["nav"] / (1 - last["fee_fraction"]) if closed else last["nav"]
        segments.append(dict(entry_date=first["date"], exit_or_mark_date=last["date"],
                             holding=first["to"], completed=closed, return_intervals=end-begin,
                             entry_reason=reasons[first["date"]],
                             return_including_exit_fee=last["nav"]/first["nav"]-1,
                             asset_return_excluding_exit_fee=gross_exit_nav/first["nav"]-1,
                             nav_contribution=last["nav"]-first["nav"]))
    switches = sum(x["from"] is not None and x["from"] != x["to"] for x in events)
    fee_factor = 1.0
    for event in events:
        fee_factor *= 1 - event["fee_fraction"]
    returns = rows[1:]
    watchdog = [line for line in read("signals/watchdog.log").splitlines() if START <= line[:10] <= END]
    watchdog_trading = [line for line in watchdog if line[:10] in actual]
    report = {
        "scope": "Committed signal-time shadow ledger, raw-price units, virtual reinvestment; not actual fills or close NAV",
        "cutoff": END,
        "canonical_date_at_audit": canonical["date"],
        "candidate_id": account["candidate_id"],
        "start": START,
        "metrics": {
            "observations": len(events), "return_intervals": len(events)-1,
            "end_nav": events[-1]["nav"], "total_return": events[-1]["nav"]-1,
            "max_drawdown": mdd, "max_drawdown_peak_trough_dates": mdd_dates,
            "current_drawdown": rows[-1]["drawdown"], "switches": switches,
            "positive_intervals": sum(x["daily_return"] > 0 for x in returns),
            "negative_intervals": sum(x["daily_return"] < 0 for x in returns),
            "best_interval": max(returns, key=lambda x: x["daily_return"]),
            "worst_interval": min(returns, key=lambda x: x["daily_return"]),
            "estimated_nav_without_booked_fees_same_holdings": events[-1]["nav"] / fee_factor,
            "booked_fee_drag_nav_points": events[-1]["nav"] / fee_factor - events[-1]["nav"],
        },
        "segments": segments,
        "comparison": strategy_comparison,
        "integrity": {
            "canonical_signal_hash_valid": True, "canonical_checksum_valid": True,
            "shadow_projection_equals_canonical": projection_checks,
            "expected_trading_dates": expected, "all_expected_dates_recorded": expected == actual,
            "all_dated_texts_date_target_rounded_nav_match": True,
            "all_events_have_no_corporate_actions": all(not x["actions"] for x in events),
        },
        "operations": {
            "watchdog_trading_day_records": len(watchdog_trading),
            "failed_records": [x for x in watchdog_trading if " FAILED:" in x],
            "zero_dual_source_comparison_dates": [x[:10] for x in watchdog_trading if "双源已完成共同历史有效对比为0" in x],
            "qvix_warning_dates": [x[:10] for x in watchdog_trading if "QVIX" in x],
            "note": "QVIX channel is disabled by V12-R2 design. Dual-source warnings limit independent data verification, not proof of wrong quotes.",
        },
        "events": events,
        "sources_sha256": sources,
        "limitations": [
            "30 signal marks, 29 return intervals and 5 completed holding segments are a small sample.",
            "Daily mark drawdown does not measure intraday drawdown.",
            "Canonical retains current account plus events; historical decision diagnostics are not all retained there.",
            "Source hashes record this audit input; live source files continue to evolve.",
            "Exit fee is charged to the ending holding segment; first entry has no modeled fee.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (OUT / "daily.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    dates = [dt.date.fromisoformat(x["date"]) for x in rows]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)
    top.plot(dates, [x["nav"] for x in rows], color="#245b8e", lw=2)
    top.axhline(1, color="gray", lw=.8, linestyle="--")
    top.set(title="V12-R2 forward shadow NAV | Sep 29 - Nov 16, 2026", ylabel="NAV (start = 1)")
    for index in starts[1:]:
        top.scatter(dates[index], rows[index]["nav"], color="#b04f30", s=26, zorder=3)
    bottom.fill_between(dates, [x["drawdown"]*100 for x in rows], 0, color="#b04f30", alpha=.3)
    bottom.set_ylabel("Drawdown (%)")
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for axes in (top, bottom):
        axes.grid(alpha=.2)
    fig.text(.5, -.015, "14:50 signal marks; 0.02% per switch; not broker fills. V9.2 / V9.2+ / V10-H paths identical.",
             ha="center", fontsize=9)
    fig.savefig(OUT / "forward_nav.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"output": str(OUT), "metrics": report["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
