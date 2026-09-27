"""Frozen V12 gates and deterministic selection; no simulation or file I/O.

Returns and drawdowns are fractions, with max_dd <= 0 (larger is better).
``full`` and ``tail`` must already describe the supplied continuous-path
training prefix. The caller computes tail from January 1 of period_end's year;
this module never substitutes another year's outcome. Optional metric dates
are checked, and future/unknown blocks are ignored rather than used as ties.
"""
from datetime import date
import math


SCENARIOS = ("close_1bp", "close_11bp", "lag1_1bp", "lag1_11bp")
STRESS_SCENARIOS = ("close_11bp", "lag1_11bp")
CONTROL_KEYS = ("v9", "v91", "v92", "simple", "h")
BLOCKS = (("early", "2014-01-02", "2017-12-31"),
          ("middle", "2018-01-01", "2021-12-31"),
          ("recent", "2022-01-01", "2025-12-31"))


def _date(value, label):
    try:
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError("%s must be an ISO date" % label)
    return value


def _mapping(values, label):
    sequence = values.values() if isinstance(values, dict) else values
    result = {}
    for record in sequence:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
            raise ValueError("%s requires records with string ids" % label)
        if record["id"] in result:
            raise ValueError("Duplicate %s id: %s" % (label, record["id"]))
        result[record["id"]] = record
    if isinstance(values, dict) and any(key != record["id"] for key, record in values.items()):
        raise ValueError("%s mapping keys must equal record ids" % label)
    return result


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("%s must be a finite number" % label)
    return float(value)


def _metrics(value, label, start, end, required):
    if not isinstance(value, dict):
        raise ValueError("Missing %s metrics" % label)
    result = {}
    for key in required:
        result[key] = _number(value.get(key), label + "." + key)
        if key == "max_dd" and not -1. <= result[key] <= 0.:
            raise ValueError("%s.max_dd must be signed in [-1, 0]" % label)
    if "start" in value and _date(value["start"], label + ".start") < start:
        raise ValueError("%s starts outside its declared period" % label)
    if "end" in value and _date(value["end"], label + ".end") > end:
        raise ValueError("%s contains observations after its declared period" % label)
    if "start" in value and "end" in value and value["start"] > value["end"]:
        raise ValueError("%s has reversed metric dates" % label)
    return result


def _pareto(ids, scores):
    """Keep all exact ties: dominance needs at least one strict improvement."""
    result = []
    for candidate in ids:
        point = scores[candidate]["pareto_coordinates"]
        dominated = False
        for other in ids:
            if other == candidate:
                continue
            comparison = scores[other]["pareto_coordinates"]
            if all(a >= b for a, b in zip(comparison, point)) and any(a > b for a, b in zip(comparison, point)):
                dominated = True
                break
        if not dominated:
            result.append(candidate)
    return result


def select(rows, candidates, controls, period_end):
    """Select only registered, selectable non-controls from a common prefix.

    rows may be records or an id->record mapping. Each record has all four
    scenarios, each with full metrics, tail metrics, blocks, and a nonnegative
    turnover_equivalent. Candidates have id/kind/selectable/complexity fields.
    Controls map v9/v91/v92/simple/h to ids; additional control ids are excluded
    from selection but do not change the five original comparison benchmarks.

    No rounding/tolerance loosens a gate. The tie-break uses precisely the two
    high-fee scenarios x available historical blocks, not lag1_1bp or full/tail.
    Missing/inconsistent required inputs stop selection instead of silently
    deleting failed trials. This function never mutates the supplied records.
    """
    period_end = _date(period_end, "period_end")
    if period_end < BLOCKS[0][1]:
        raise ValueError("period_end precedes the fixed research start")
    active_blocks = [(name, start, min(end, period_end)) for name, start, end in BLOCKS
                     if start <= period_end]
    block_names = [block[0] for block in active_blocks]
    records, configs = _mapping(rows, "rows"), _mapping(candidates, "candidates")
    if set(records) != set(configs):
        raise ValueError("rows and candidates must have exactly the same ids")
    if not isinstance(controls, dict) or any(key not in controls for key in CONTROL_KEYS):
        raise ValueError("All five original control ids are required")
    if any(not isinstance(cid, str) or cid not in records for cid in controls.values()):
        raise ValueError("Every control id must have a row and candidate")
    if len({controls[key] for key in CONTROL_KEYS}) != len(CONTROL_KEYS):
        raise ValueError("The five original controls must have distinct ids")
    control_ids = set(controls.values())
    tail_start = period_end[:4] + "-01-01"
    data = {}
    selectable = []
    for cid in sorted(records):
        config, row = configs[cid], records[cid]
        if config.get("kind") not in ("single", "allocation", "benchmark"):
            raise ValueError("Unknown candidate kind: " + cid)
        if not isinstance(config.get("selectable"), bool):
            raise ValueError("selectable must be explicit: " + cid)
        complexity = config.get("complexity")
        if isinstance(complexity, bool) or not isinstance(complexity, int) or complexity < 0:
            raise ValueError("complexity must be a nonnegative integer: " + cid)
        if config["selectable"] and config["kind"] != "benchmark" and cid not in control_ids:
            selectable.append(cid)
        if "period_end" in row and row["period_end"] != period_end:
            raise ValueError("Row period_end does not match selection: " + cid)
        scenarios = row.get("scenarios", {})
        data[cid] = {}
        for name in SCENARIOS:
            scenario = scenarios.get(name)
            if not isinstance(scenario, dict):
                raise ValueError("Missing scenario: %s/%s" % (cid, name))
            label = cid + "/" + name
            full = _metrics(scenario.get("full"), label + "/full", BLOCKS[0][1], period_end, ("cagr", "max_dd"))
            tail = _metrics(scenario.get("tail"), label + "/tail", tail_start, period_end, ("total_return",))
            blocks = {}
            for block, start, end in active_blocks:
                blocks[block] = _metrics(scenario.get("blocks", {}).get(block), label + "/" + block,
                                         start, end, ("cagr",))
            turnover = _number(scenario.get("turnover_equivalent"), label + "/turnover_equivalent")
            if turnover < 0.:
                raise ValueError("turnover_equivalent must be nonnegative: " + label)
            data[cid][name] = dict(full=full, tail=tail, blocks=blocks, turnover_equivalent=turnover)

    reference = data[controls["h"]]
    v92 = data[controls["v92"]]
    baselines = {}
    for name in SCENARIOS:
        baselines[name] = dict(
            best_cagr=max(data[controls[key]][name]["full"]["cagr"] for key in CONTROL_KEYS),
            best_drawdown=max(data[controls[key]][name]["full"]["max_dd"] for key in CONTROL_KEYS))
    checks, scores = {}, {}
    for cid in sorted(records):
        scenarios = data[cid]
        full, tail = scenarios["close_1bp"]["full"], scenarios["close_1bp"]["tail"]
        excess_cagr = full["cagr"] - reference["close_1bp"]["full"]["cagr"]
        improvement_dd = full["max_dd"] - baselines["close_1bp"]["best_drawdown"]
        excess_tail = tail["total_return"] - v92["close_1bp"]["tail"]["total_return"]
        checks[cid] = dict(
            main_cagr=full["cagr"] >= reference["close_1bp"]["full"]["cagr"] + .01,
            main_drawdown=full["max_dd"] >= baselines["close_1bp"]["best_drawdown"] + .005,
            tail_return=tail["total_return"] >= v92["close_1bp"]["tail"]["total_return"])
        stress_excess = {}
        for name in STRESS_SCENARIOS:
            checks[cid][name + "_cagr"] = scenarios[name]["full"]["cagr"] >= baselines[name]["best_cagr"]
            checks[cid][name + "_drawdown"] = scenarios[name]["full"]["max_dd"] >= baselines[name]["best_drawdown"]
            for block in block_names:
                stress_excess[name + "/" + block] = (scenarios[name]["blocks"][block]["cagr"] -
                                                       reference[name]["blocks"][block]["cagr"])
        for block in block_names:
            checks[cid][block + "_cagr"] = (scenarios["close_1bp"]["blocks"][block]["cagr"] >=
                                               reference["close_1bp"]["blocks"][block]["cagr"] - .05)
        standardized = dict(cagr=excess_cagr / .02, drawdown=improvement_dd / .02, tail=excess_tail / .10)
        scores[cid] = dict(
            pareto_coordinates=[full["cagr"], full["max_dd"], tail["total_return"]],
            standardized_margins=standardized, min_margin=min(standardized.values()),
            cagr_excess_h=excess_cagr, drawdown_improvement_best=improvement_dd, tail_excess_v92=excess_tail,
            worst_stress_block_excess=min(stress_excess.values()), stress_block_excess=stress_excess,
            complexity=configs[cid]["complexity"], turnover_equivalent=scenarios["close_1bp"]["turnover_equivalent"])

    def ranking(cid):
        score = scores[cid]
        return (-score["min_margin"], -score["worst_stress_block_excess"], score["complexity"],
                score["turnover_equivalent"], cid)

    qualified = [cid for cid in selectable if all(checks[cid].values())]
    pareto = sorted(_pareto(qualified, scores), key=ranking)
    exploratory = {}
    for label, coordinate in (("top_return", 0), ("least_drawdown", 1), ("top_tail", 2)):
        exploratory[label] = min(selectable, key=lambda cid: (-scores[cid]["pareto_coordinates"][coordinate], cid)) if selectable else None
    return dict(
        primary=pareto[0] if pareto else None, qualified_count=len(qualified),
        qualified_ids=sorted(qualified, key=ranking), pareto_ids=pareto,
        selectable_count=len(selectable), checks=checks, scores=scores, controls=dict(controls), baselines=baselines,
        selection_period=[BLOCKS[0][1], period_end], tail_period=[tail_start, period_end], available_blocks=block_names,
        stress_scenarios=list(STRESS_SCENARIOS), ranking_order=["pareto", "min_margin_desc", "worst_stress_block_excess_desc",
                                                             "complexity_asc", "turnover_equivalent_asc", "id_asc"],
        status="qualified_historical_candidate" if pareto else "no_qualified_candidate",
        clean_oos=False, **exploratory)
