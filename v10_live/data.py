"""Live raw-price to total-return extension; never alter research/legacy data.

The only persistent output is data/v10_live/completed.json. Today's quote and
bar are provisional and never become a completed anchor during this call.
Corporate-action inference is conditional on the vendor's affine adjustment.
Unknown share conversions, inconsistent quotes and incomplete histories fail
closed. Output TR O/H/L are placeholders, not executable prices.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from bisect import bisect_left, bisect_right
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import urllib.request


CODES = ("159915", "588080", "510300", "510500", "563300", "512400", "512890",
         "513100", "513120", "518880", "511880")
DAY_ALIAS_CODES = frozenset(("563300", "513120"))
BENCHMARK = "510300"
ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = ROOT / "v10_h_close"
CACHE_DIR = ROOT / "data/v10_live"
TZ = timezone(timedelta(hours=8))
MAX_AGE_SECONDS = 180
MAX_QUOTE_SKEW_SECONDS = 60
REQUEST_TIMEOUT = 4
NETWORK_DEADLINE = 42
TICK = .001
EPS = TICK / 2
TOL = 1e-9


class LiveDataError(ValueError):
    pass


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _date(value):
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError):
        raise LiveDataError("非法交易日期")
    if parsed.strftime("%Y-%m-%d") != value:
        raise LiveDataError("非法交易日期")
    return parsed


def _number(value, label, positive=False):
    try:
        n = float(value)
    except (TypeError, ValueError):
        raise LiveDataError(label + "缺失或非法")
    if not math.isfinite(n) or (n <= 0 if positive else n < 0):
        raise LiveDataError(label + "缺失或非法")
    return n


def _clock(now):
    now = datetime.now(TZ) if now is None else now
    return now.replace(tzinfo=TZ) if now.tzinfo is None else now.astimezone(TZ)


def _quotes(quotes, signal_date, now, observe=False):
    moment = _clock(now)
    if not observe and (moment.date().isoformat() != signal_date or _date(signal_date).weekday() >= 5):
        raise LiveDataError("V10实时日期不是当前交易工作日")
    stamps, out = [], {}
    for code in CODES:
        q = quotes.get(code)
        if not isinstance(q, dict) or q.get("date") != signal_date:
            raise LiveDataError(code + "缺少当前交易日报价")
        try:
            stamp = datetime.strptime(q["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ)
        except (KeyError, TypeError, ValueError):
            raise LiveDataError(code + "报价时间戳非法")
        age = (moment - stamp).total_seconds()
        # observe 复盘：时钟/日期与新鲜度断言豁免，报价日期必须等于信号日不变
        if stamp.date().isoformat() != signal_date or (not observe and not -5 <= age <= MAX_AGE_SECONDS):
            raise LiveDataError(code + "报价已陈旧或来自未来")
        row = dict(price=_number(q.get("price"), code + "价格", True),
                   open=_number(q.get("open"), code + "开盘", True),
                   volume=_number(q.get("volume"), code + "累计成交量", True),
                   timestamp=q["timestamp"])
        for field in ("prev_close", "high", "low"):
            if q.get(field) is not None:
                row[field] = _number(q[field], code + field, True)
        if "high" in row and "low" in row:
            if row["low"] > min(row["open"], row["price"]) + TICK or row["high"] < max(row["open"], row["price"]) - TICK:
                raise LiveDataError(code + "报价OHLC不相容")
        stamps.append(stamp)
        out[code] = row
    if (max(stamps) - min(stamps)).total_seconds() > MAX_QUOTE_SKEW_SECONDS:
        raise LiveDataError("V10资产报价时间差过大")
    return out


def _rows(values, end, adjusted=False):
    out = []
    for row in values:
        if len(row) != 4:
            raise LiveDataError("行情行必须为date/open/close/volume四列")
        date = row[0]
        _date(date)
        if date > end or (out and date <= out[-1][0]):
            raise LiveDataError("行情日期回退、重复或越界")
        try:
            o, c, v = map(float, row[1:])
        except (ValueError, TypeError):
            raise LiveDataError("日K字段非法")
        if not all(math.isfinite(x) for x in (o, c, v)) or v < 0 or (not adjusted and min(o, c) <= 0):
            raise LiveDataError("日K价格或量非法")
        out.append((date, o, c, v))
    if not out:
        raise LiveDataError("日K响应为空")
    return out


def _load_seed():
    path = SEED_DIR / "corrected_manifest.json"
    manifest = json.loads(path.read_text())
    raw, tr = {}, {}
    for code in CODES:
        asset = manifest["assets"][code]
        rp = SEED_DIR / "results/price_audit" / (code + "_raw_2010-01-01_" + manifest["end"] + ".json")
        cp = SEED_DIR / "corrected_snapshots" / (code + ".csv")
        if _file_hash(rp) != asset["raw_sha256"] or _file_hash(cp) != asset["sha256"]:
            raise LiveDataError(code + "冻结种子哈希变化")
        raw[code] = _rows([(r[0], r[1], r[2], r[5]) for r in json.loads(rp.read_text())["rows"]], manifest["end"])
        with cp.open(newline="") as f:
            tr[code] = _rows(list(csv.reader(f)), manifest["end"])
        if [r[0] for r in raw[code]] != [r[0] for r in tr[code]] or raw[code][-1][0] != manifest["end"]:
            raise LiveDataError(code + "冻结种子日期轴不一致")
    return dict(end=manifest["end"], manifest_hash=_file_hash(path), raw=raw, tr=tr,
                assets=manifest["assets"])


def _empty_state(seed):
    return dict(schema=2, seed_manifest_sha256=seed["manifest_hash"], seed_end=seed["end"],
                last_completed=seed["end"], calendar=[],
                assets={c: dict(raw=[], tr=[], actions={}, last_observed_completed=seed["end"])
                        for c in CODES})


def _is_no_action_gap(action):
    return (action.get("not_observed") is True
            and action.get("verification") == "bracketed_no_action_interval"
            and action.get("split_ratio") == 1
            and action.get("cash_per_old_share") == 0)


def _read_state(path, seed):
    if not path.exists():
        return _empty_state(seed)
    envelope = json.loads(path.read_text())
    state = envelope["state"]
    if envelope.get("sha256") != _hash(state) or state.get("schema") not in (1, 2):
        raise LiveDataError("V10独立缓存校验失败")
    if state.get("seed_manifest_sha256") != seed["manifest_hash"] or set(state["assets"]) != set(CODES):
        raise LiveDataError("V10缓存与冻结种子不一致")
    old_schema = state["schema"] == 1
    if old_schema:
        # Schema 1 had exactly matching asset calendars. Migrate only that
        # valid contract; never reinterpret a damaged old cache as a gap.
        state["calendar"] = [r[0] for r in state["assets"][BENCHMARK]["raw"]]
    calendar = state["calendar"]
    if (calendar != sorted(set(calendar))
            or (calendar and (calendar[0] <= seed["end"] or calendar[-1] != state["last_completed"]))
            or (not calendar and state["last_completed"] != seed["end"])):
        raise LiveDataError("V10缓存基准日历非法")
    for day in calendar:
        _date(day)
    for code in CODES:
        asset = state["assets"][code]
        ds = [r[0] for r in asset["raw"]]
        if (ds != sorted(set(ds)) or ds != [r[0] for r in asset["tr"]]
                or set(asset["actions"]) != set(calendar) or not set(ds).issubset(calendar)):
            raise LiveDataError(code + "缓存缺少已确认行情或行动条目")
        if (old_schema or code == BENCHMARK) and ds != calendar:
            raise LiveDataError(code + "旧缓存/基准日期不完整")
        if ds:
            _rows(asset["raw"], state["last_completed"])
            _rows(asset["tr"], state["last_completed"])
        last_observed = ds[-1] if ds else seed["end"]
        if old_schema:
            asset["last_observed_completed"] = last_observed
        if (asset.get("last_observed_completed") != last_observed
                or (ds and ds[0] <= seed["end"])):
            raise LiveDataError(code + "缓存日期非法")
        observed = set(ds)
        for day, action in asset["actions"].items():
            if day not in observed:
                if (not _is_no_action_gap(action)
                        or not action.get("previous_quote_date", "") < day < action.get("next_quote_date", "")):
                    raise LiveDataError(code + "缺报价日没有明确的跨期无行动核验")
            elif action.get("not_observed"):
                raise LiveDataError(code + "缓存缺日标记与真实报价冲突")
    state["schema"] = 2
    return state


def _write_state(path, state):
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent),
                                         prefix=".completed-", suffix=".tmp", delete=False) as f:
            name = f.name
            json.dump(dict(state=state, sha256=_hash(state)), f, ensure_ascii=False, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, str(path))
        name = None
    finally:
        if name is not None:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass


def _provider_quote_info(node, symbol, code, label):
    """Keep only the public quote fields needed to check an adjustment anchor."""
    qt = (node.get("qt") or {}).get(symbol) or []
    if not qt:
        return None
    try:
        stamp = datetime.strptime(str(qt[30]), "%Y%m%d%H%M%S")
        return dict(date=stamp.strftime("%Y-%m-%d"),
                    timestamp=stamp.strftime("%Y-%m-%d %H:%M:%S"),
                    prev_close=_number(qt[4], code + label + "前收", True),
                    open=_number(qt[5], code + label + "开盘", True))
    except (IndexError, TypeError, ValueError):
        raise LiveDataError(code + label + "响应的qt锚点非法")


def _fetch_pair(code, start, end):
    symbol = ("sh" if code.startswith("5") else "sz") + code
    result, urls = {}, {}
    for label, mode, key in (("raw", "", "day"), ("qfq", "qfq", "qfqday")):
        url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param="
               + "%s,day,%s,%s,640,%s" % (symbol, start, end, mode))
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            body = response.read(4000001)
        if len(body) > 4000000:
            raise LiveDataError(code + "日K响应过大")
        node = (json.loads(body.decode("utf-8")).get("data") or {}).get(symbol) or {}
        result[label + "_provider_version"] = str(node.get("version"))
        result[label + "_quote_info"] = _provider_quote_info(node, symbol, code, label)
        page = node.get(key)
        if label == "qfq" and not page and code in DAY_ALIAS_CODES and node.get("day"):
            # Observed Tencent v16 representation for these two zero-action
            # funds. This is only a candidate alias; _validate_day_alias must
            # check action history, all completed O/C and the current qt anchor.
            page = node["day"]
            if result["qfq_quote_info"] is None:
                raise LiveDataError(code + "未调整day响应缺可核对的qt锚点")
            result["qfq_alias"] = "day"
        if not page:
            raise LiveDataError(code + "缺少明确的" + key + "行情")
        try:
            result[label] = [(r[0], r[1], r[2], r[5]) for r in page]
        except (IndexError, TypeError):
            raise LiveDataError(code + "日K字段不完整")
        urls[label] = url
    result["sources"] = urls
    return result


def _current_anchor_times(code, pair, quote, signal_date, moment, observe=False):
    """The delayed-qfq bridge needs two fresh, contemporaneous qt anchors."""
    stamps = []
    for label in ("raw", "qfq"):
        qt = pair.get(label + "_quote_info") or {}
        try:
            stamp = datetime.strptime(qt["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ)
        except (KeyError, TypeError, ValueError):
            raise LiveDataError(code + "缺少当日qfq桥接所需的双端qt时间戳")
        if (qt.get("date") != signal_date or stamp.date().isoformat() != signal_date
                or (not observe and not -5 <= (_clock(moment) - stamp).total_seconds() <= MAX_AGE_SECONDS)):
            raise LiveDataError(code + "当日qfq桥接qt已陈旧或来自未来")
        stamps.append(stamp)
    stamps.append(datetime.strptime(quote["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ))
    if (max(stamps) - min(stamps)).total_seconds() > MAX_QUOTE_SKEW_SECONDS:
        raise LiveDataError(code + "当日qfq桥接双端qt与主报价时间差过大")


def _complete_current_qfq(code, pair, raw, qfq, quote, seed, state,
                          splits, signal_date, moment):
    """Bridge only an unpublished current OPEN under zero-detected-action checks.

    Tencent v18 can expose completed qfqday rows and a current qt while its raw
    v16 day already includes today. This does not provide a published qfq close.
    A neutral last completed adjustment and unchanged, fresh ex-reference allow
    a provisional zero-action opening anchor at the vendor's tick precision.
    It is not issuer-level proof against sub-tick actions. Any detectable or
    declared new cash/split, multiple absent rows or uncertain quote is refused.
    The returned anchor must never be persisted as a completed qfq observation.
    """
    if [r[0] for r in raw] == [r[0] for r in qfq]:
        return qfq, None
    if (pair.get("qfq_alias") is not None
            or pair.get("raw_provider_version") != "16"
            or pair.get("qfq_provider_version") != "18"
            or len(raw) < 2 or raw[-1][0] != signal_date
            or [r[0] for r in qfq] != [r[0] for r in raw[:-1]]):
        raise LiveDataError(code + "当前日配对raw/qfq尚未确认或日期缺失")
    _current_anchor_times(code, pair, quote, signal_date, moment)
    known_cash = [e for e in seed["assets"][code]["cash_events"] if e["date"] == signal_date]
    saved_action = state["assets"][code]["actions"].get(signal_date, {})
    if (splits.get(signal_date, Decimal(1)) != 1
            or any(e.get("cash_per_old_share", 0) != 0 for e in known_cash)
            or saved_action.get("cash_per_old_share", 0) != 0
            or saved_action.get("split_ratio", 1) != 1):
        raise LiveDataError(code + "当前有已知现金/折算行动，禁止桥接缺失qfq")
    if any(abs(raw[-2][k] - qfq[-1][k]) > EPS + TOL for k in (1, 2)):
        raise LiveDataError(code + "末个完成日复权偏移非中性，不能确认当前无新行动")
    previous, opening = raw[-2][2], raw[-1][1]
    if (abs(_number(quote.get("prev_close"), code + "桥接主报价前收", True) - previous) > EPS + TOL
            or abs(quote["open"] - opening) > EPS + TOL):
        raise LiveDataError(code + "桥接主报价前收/开盘与raw锚点不一致")
    for label in ("raw", "qfq"):
        qt = pair[label + "_quote_info"]
        if (abs(_number(qt.get("prev_close"), code + label + "桥接前收", True) - previous) > EPS + TOL
                or abs(_number(qt.get("open"), code + label + "桥接开盘", True) - opening) > EPS + TOL):
            raise LiveDataError(code + "桥接双端qt前收/开盘不能确认无新行动")
    details = dict(date=signal_date, verification="no_detected_action_within_tick",
                   price_tick=TICK, observed_raw_open=opening,
                   previous_raw_close=previous, last_published_qfq_date=qfq[-1][0],
                   synthetic_provisional=True, published_qfq_close=False,
                   meaning="observed raw current open used only as a provisional qfq open anchor; no published qfq close",
                   limitation="conditional on vendor ex-reference and affine adjustment, not issuer-confirmed zero cash")
    # The close column is deliberately an OPEN placeholder. _infer_actions uses
    # only today's open, and the actual TR mark/volume come from the main quote.
    return list(qfq) + [(signal_date, opening, opening, raw[-1][3])], details


def _validate_day_alias(code, pair, raw, qfq, seed, state, overrides, signal_date):
    if pair.get("qfq_alias") is None:
        return False
    if code not in DAY_ALIAS_CODES or pair.get("qfq_alias") != "day" or pair.get("qfq_provider_version") != "16":
        raise LiveDataError(code + "未验证的qfq day响应别名")
    asset = seed["assets"][code]
    prior = state["assets"][code]["actions"].values()
    if (asset["cash_events"] or asset["split_events"] or (overrides or {}).get(code)
            or any(a["cash_per_old_share"] != 0 or a["split_ratio"] != 1 for a in prior)):
        raise LiveDataError(code + "已有或新声明公司行动，禁止使用未调整day响应")
    if len(raw) < 2 or any(any(abs(r[k] - q[k]) > TOL for k in (1, 2))
                           for r, q in zip(raw[:-1], qfq[:-1])):
        raise LiveDataError(code + "未调整别名与完成日raw价格不一致")
    qt = pair.get("qfq_quote_info") or {}
    if (qt.get("date") != signal_date
            or abs(_number(qt.get("prev_close"), code + "别名前收", True) - raw[-2][2]) > EPS + TOL
            or abs(_number(qt.get("open"), code + "别名开盘", True) - raw[-1][1]) > EPS + TOL):
        raise LiveDataError(code + "未调整别名的当前qt/前收/开盘不能确认无调整")
    return True


def _fetch_all(start, end, fetcher):
    executor = ThreadPoolExecutor(max_workers=3)
    futures = {executor.submit(fetcher, c, start, end): c for c in CODES}
    out = {}
    try:
        for future in as_completed(futures, timeout=NETWORK_DEADLINE):
            code = futures[future]
            out[code] = future.result()
        return out
    except TimeoutError:
        raise LiveDataError("V10配对日K获取超过42秒期限")
    except Exception as error:
        if isinstance(error, LiveDataError):
            raise
        raise LiveDataError("V10配对日K获取失败: " + type(error).__name__)
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False)


def _split_map(code, seed, state, overrides):
    known = {}
    for event in seed["assets"][code]["split_events"]:
        known[event["date"]] = Decimal(str(event.get("ratio_exact", event["ratio"])))
    for date, event in state["assets"][code]["actions"].items():
        if event["split_ratio"] != 1:
            known[date] = Decimal(str(event.get("split_ratio_exact", event["split_ratio"])))
    for date, value in (overrides or {}).get(code, {}).items():
        _date(date)
        ratio = Decimal(str(value.get("ratio_exact", value["ratio"]) if isinstance(value, dict) else value))
        if not ratio.is_finite() or ratio <= 0:
            raise LiveDataError(code + "显式折算比例非法")
        if date in known and ratio != known[date]:
            raise LiveDataError(code + "显式折算比例与已确认行动冲突")
        known[date] = ratio
    return known


def _infer_actions(code, raw, qfq, splits, quote, signal_date):
    dates = [r[0] for r in raw]
    scales, future = [0.] * len(raw), Decimal(1)
    for i in range(len(raw) - 1, -1, -1):
        scales[i] = float(Decimal(1) / future)
        if dates[i] in splits:
            future *= splits[dates[i]]
    if abs(raw[-1][1] - quote["open"]) > EPS + TOL:
        raise LiveDataError(code + "当前日K开盘与实时报价不一致")
    if abs(qfq[-1][1] - raw[-1][1]) > EPS + TOL:
        raise LiveDataError(code + "当前qfq开盘不是可确认的原始价格基准")
    intervals, points = [], []
    no_split = all(abs(a - 1) < 1e-12 for a in scales)
    for i, (r, q, a) in enumerate(zip(raw, qfq, scales)):
        opening = q[1] - a * r[1]
        if r[0] == signal_date:
            # Stable opens only: qfq and raw intraday closes can be captured at
            # different instants and must never manufacture a dividend.
            offsets = [opening]
        else:
            offsets = [opening, q[2] - a * r[2]]
            allowed = TOL if no_split else TICK + TOL
            if max(offsets) - min(offsets) > allowed:
                raise LiveDataError(code + "出现未知折算或raw/qfq仿射不一致: " + r[0])
        low, high = max(offsets) - EPS, min(offsets) + EPS
        intervals.append((low, high))
        points.append(sum(offsets) / len(offsets))
    feasible, actions = intervals[0], {}
    for i in range(1, len(raw)):
        cash = lower = upper = 0.
        if no_split:
            jump = points[i] - points[i - 1]
            if jump < -TOL:
                raise LiveDataError(code + "复权出现无法解释的负现金变化")
            if jump > TOL:
                cash, lower, upper = jump, max(0., jump - TICK), jump + TICK
        else:
            lo, hi = max(feasible[0], intervals[i][0]), min(feasible[1], intervals[i][1])
            if lo <= hi + TOL:
                feasible = (lo, hi) if lo <= hi else ((lo + hi) / 2,) * 2
            elif intervals[i][1] < feasible[0]:
                raise LiveDataError(code + "折算后的现金区间异常")
            else:
                jump = (sum(intervals[i]) - sum(feasible)) / 2
                if jump < TICK - TOL:
                    raise LiveDataError(code + "现金/折算变化低于可确认精度")
                cash = jump / scales[i - 1]
                lower = max(0., (intervals[i][0] - feasible[1]) / scales[i - 1])
                upper = (intervals[i][1] - feasible[0]) / scales[i - 1]
                feasible = intervals[i]
        if cash / raw[i - 1][2] > .10:
            raise LiveDataError(code + "推断现金超过前收10%，需独立核验分红/折算")
        ratio = scales[i] / scales[i - 1]
        actions[dates[i]] = dict(split_ratio=ratio, split_ratio_exact=str(splits.get(dates[i], Decimal(1))),
                                cash_per_old_share=cash, cash_lower=lower, cash_upper=upper,
                                previous_quote_date=dates[i - 1],
                                verification="same_vintage_affine_checks",
                                split_source="explicit_or_previously_verified" if ratio != 1 else "unit_slope_consistent",
                                cash_source="vendor_affine_inference" if cash else "no_detected_cash_change")
    return actions


def _gap_actions(code, raw, actions, calendar, splits, seed_end):
    """Annotate absent benchmark sessions without manufacturing a price bar.

    An affine jump spanning absent observations cannot locate a cash ex-date.
    Such intervals remain blocked. An explicitly dated conversion at the next
    quoted bar is already located by the supplied split schedule; otherwise a
    recoverable interval must have both zero cash and unchanged share units.
    """
    result = {}
    for date, action in actions.items():
        if date <= seed_end:
            continue
        previous = action["previous_quote_date"]
        missing = calendar[bisect_right(calendar, previous):bisect_left(calendar, date)]
        missing = [day for day in missing if day > seed_end]
        if not missing:
            continue
        located_split = date in splits and math.isclose(
            action["split_ratio"], float(splits[date]), rel_tol=1e-12, abs_tol=1e-12)
        if action["cash_per_old_share"] != 0 or (action["split_ratio"] != 1 and not located_split):
            raise LiveDataError(code + "跨缺报价区间存在无法定位的现金/拆分行动，需独立核验: "
                                + previous + "至" + date)
        for day in missing:
            result[day] = dict(not_observed=True, split_ratio=1., split_ratio_exact="1",
                               cash_per_old_share=0., cash_lower=0., cash_upper=0.,
                               verification="bracketed_no_action_interval",
                               previous_quote_date=previous, next_quote_date=date,
                               cash_source="no_cash_change_between_observed_quotes",
                               split_source="no_declared_unit_change_on_missing_session")
    return result


def _build(seed, state, pairs, quotes, signal_date, overrides, validation_time=None):
    checked, provisional_anchors = {}, {}
    known_raw = {c: seed["raw"][c] + [tuple(r) for r in state["assets"][c]["raw"]] for c in CODES}
    for code in CODES:
        raw = _rows(pairs[code]["raw"], signal_date)
        qfq = _rows(pairs[code]["qfq"], signal_date, adjusted=True)
        splits = _split_map(code, seed, state, overrides)
        qfq, anchor = _complete_current_qfq(code, pairs[code], raw, qfq, quotes[code],
                                            seed, state, splits, signal_date, validation_time)
        provisional_anchors[code] = anchor
        if [r[0] for r in raw] != [r[0] for r in qfq] or raw[-1][0] != signal_date:
            raise LiveDataError(code + "当前日配对raw/qfq尚未确认或日期缺失")
        _validate_day_alias(code, pairs[code], raw, qfq, seed, state, overrides, signal_date)
        previous = {r[0]: r for r in known_raw[code]}
        overlap = [r for r in raw if r[0] in previous]
        last_observed = known_raw[code][-1][0]
        if len(overlap) < 5 or last_observed not in {r[0] for r in overlap}:
            raise LiveDataError(code + "完成日线缺少足够重叠锚点")
        expected = {d for d in previous if raw[0][0] <= d <= state["last_completed"]}
        if expected - {r[0] for r in overlap}:
            raise LiveDataError(code + "已知历史日期在新响应中缺失")
        for r in overlap:
            if any(abs(r[k] - previous[r[0]][k]) > TOL for k in (1, 2, 3)):
                raise LiveDataError(code + "历史raw被修订，不能静默重写TR种子或锚点")
        if any(known_raw[code][0][0] <= r[0] <= state["last_completed"] and r[0] not in previous for r in raw):
            raise LiveDataError(code + "已记录缺报价日被供应商补回，需显式核验后重建，不能静默改写历史")
        if any(raw[0][0] < d <= signal_date and d not in {r[0] for r in raw} for d in splits):
            raise LiveDataError(code + "折算override不是首个新单位报价日")
        actions = _infer_actions(code, raw, qfq, splits, quotes[code], signal_date)
        if anchor is not None:
            if (actions[signal_date]["cash_per_old_share"] != 0
                    or actions[signal_date]["split_ratio"] != 1):
                raise LiveDataError(code + "桥接qfq推断出非零行动，拒绝继续")
            actions[signal_date]["current_qfq_open_anchor_provisional"] = True
            actions[signal_date]["verification"] = "no_detected_action_within_tick"
        old_actions = {e["date"]: e for e in seed["assets"][code]["cash_events"]}
        for date, action in actions.items():
            if date <= seed["end"]:
                old_cash = old_actions.get(date, {}).get("cash_per_old_share", 0.)
            elif date <= state["last_completed"]:
                old_cash = state["assets"][code]["actions"][date]["cash_per_old_share"]
            else:
                continue
            if not action["cash_lower"] - TOL <= old_cash <= action["cash_upper"] + TOL:
                raise LiveDataError(code + "新复权版本与已确认现金行动不一致")
        tail_dates = [r[0] for r in raw if r[0] > state["last_completed"]]
        checked[code] = (raw, actions, tail_dates, splits)
    calendar_tail = checked[BENCHMARK][2]
    if not calendar_tail or calendar_tail[-1] != signal_date:
        raise LiveDataError("缺少当前基准交易日")
    calendar = [r[0] for r in known_raw[BENCHMARK]] + calendar_tail
    if any(set(checked[c][2]) - set(calendar_tail) for c in CODES):
        raise LiveDataError("资产报价日超出基准日历，需核验基准是否缺行情")
    next_state = deepcopy(state)
    next_state["schema"] = 2
    next_state["calendar"] = list(state.get("calendar", [r[0] for r in state["assets"][BENCHMARK]["raw"]])) + calendar_tail[:-1]
    histories, raw_histories, output_actions, receipts = {}, {}, {}, {}
    for code in CODES:
        raw, inferred, _, splits = checked[code]
        gaps = _gap_actions(code, raw, inferred, calendar, splits, seed["end"])
        by_date = {r[0]: r for r in raw}
        asset = next_state["assets"][code]
        base_tr = seed["tr"][code] + [tuple(r) for r in asset["tr"]]
        raw_rows = list(known_raw[code])
        all_actions = deepcopy(asset["actions"])
        index, previous_raw = base_tr[-1][2], raw_rows[-1][2]
        cumulative_split = 1.
        for saved_action in asset["actions"].values():
            cumulative_split *= saved_action["split_ratio"]
        for date in calendar_tail:
            if date not in by_date:
                if date not in gaps:
                    raise LiveDataError(code + "缺报价区间没有前后无行动确认: " + date)
                # No artificial close, volume, dividend reinvestment or trade.
                all_actions[date] = gaps[date]
                asset["actions"][date] = gaps[date]
                continue
            action = inferred[date]
            daily = by_date[date]
            current = date == signal_date
            price = quotes[code]["price"] if current else daily[2]
            opening = quotes[code]["open"] if current else daily[1]
            volume = quotes[code]["volume"] if current else daily[3]
            ratio, cash = action["split_ratio"], action["cash_per_old_share"]
            if current and "prev_close" in quotes[code]:
                reference = (previous_raw - cash) / ratio
                if min(abs(quotes[code]["prev_close"] - previous_raw),
                       abs(quotes[code]["prev_close"] - reference)) > TICK + TOL:
                    raise LiveDataError(code + "报价前收与raw锚点/除权参考均不符")
            index *= (ratio * price + cash) / previous_raw
            cumulative_split *= ratio
            tr_row = (date, index, index, volume / cumulative_split)
            raw_row = (date, opening, price, volume)
            if not math.isfinite(index) or index <= 0:
                raise LiveDataError(code + "TR延展产生非法值")
            base_tr.append(tr_row)
            raw_rows.append(raw_row)
            all_actions[date] = action
            if not current:
                asset["raw"].append(raw_row)
                asset["tr"].append(tr_row)
                asset["actions"][date] = action
            previous_raw = price
        asset["last_observed_completed"] = asset["raw"][-1][0] if asset["raw"] else seed["end"]
        histories[code] = [(d, o, c, c, c, v) for d, o, c, v in base_tr]
        raw_histories[code] = raw_rows
        output_actions[code] = all_actions
        receipts[code] = dict(raw_response_sha256=_hash(pairs[code]["raw"]),
                              qfq_response_sha256=_hash(pairs[code]["qfq"]),
                              source_urls=pairs[code].get("sources", {}),
                              qfq_representation=("explicit_completed_qfqday_with_provisional_current_open" if provisional_anchors[code]
                                                  else "validated_no_known_action_day_alias" if pairs[code].get("qfq_alias") else "explicit_qfqday"),
                              raw_quote_info=pairs[code].get("raw_quote_info"),
                              qfq_quote_info=pairs[code].get("qfq_quote_info"),
                              current_qfq_open_anchor_provisional=provisional_anchors[code])
    next_state["last_completed"] = calendar_tail[-2] if len(calendar_tail) > 1 else state["last_completed"]
    return dict(histories=histories, raw_histories=raw_histories, actions=output_actions,
                calendar=calendar,
                metadata=dict(seed_end=seed["end"], seed_manifest_sha256=seed["manifest_hash"],
                              completed_through=next_state["last_completed"], signal_date=signal_date,
                              provisional_date=signal_date, data_view="independent_live_TR_close",
                              quote_timestamps={c: quotes[c]["timestamp"] for c in CODES},
                              receipts=receipts, volume_unit="fixed_seed_end_share_units",
                              last_observed_completed={c: next_state["assets"][c]["last_observed_completed"] for c in CODES},
                              not_observed={c: [d for d, a in output_actions[c].items() if a.get("not_observed")]
                                            for c in CODES},
                              open_high_low="TR close placeholders; raw prices are separate",
                              usable_for_next_open=False,
                              calendar_source="fresh benchmark raw day sequence; no inferred asset fills",
                              cash_inference="conditional affine reconstruction with reported amount bounds; not all issuer-confirmed",
                              provisional_mark="prior completed TR * (split * current raw price + cash) / prior raw close")), next_state


def build_live_view(quotes, signal_date, now=None, *, split_overrides=None,
                    cache_dir=None, fetch_pair=None, observe=False):
    """Return independent H histories/raw_histories/actions/calendar/metadata.

    ``fetch_pair(code,start,end)`` is injectable and returns raw/qfq four-column
    rows. ``split_overrides={code:{first_new_unit_date:ratio_or_ratio_dict}}``
    explicitly supplies independently checked new/old share ratios.
    Every validated post-seed date has an action entry, including zero events.
    This function never prints, sends a message, or mutates a legacy cache.
    observe=True exempts only the wall-clock/freshness assertions (read-only
    estimates replaying a recent session); every data-consistency check stays.
    """
    _date(signal_date)
    started = time.monotonic()
    reference_clock = _clock(now)
    normalized_quotes = _quotes(quotes, signal_date, reference_clock, observe=observe)
    if set(split_overrides or {}) - set(CODES):
        raise LiveDataError("折算override含非H池资产")
    seed = _load_seed()
    if signal_date <= seed["end"]:
        raise LiveDataError("实时日期必须晚于冻结种子；禁止将冻结回测冒充实时")
    directory = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "update.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LiveDataError("V10独立数据更新正在进行，请稍后重试")
        state_path = directory / "completed.json"
        state = _read_state(state_path, seed)
        if state["last_completed"] >= signal_date:
            raise LiveDataError("缓存时间晚于实时请求，禁止倒退或重用历史信号")
        if (_date(signal_date) - _date(state["last_completed"])).days > 600:
            raise LiveDataError("V10增量超过单页安全范围，需独立补齐历史")
        observed_rows = {c: seed["raw"][c] + state["assets"][c]["raw"] for c in CODES}
        if any((_date(signal_date) - _date(rows[-1][0])).days > 600 for rows in observed_rows.values()):
            raise LiveDataError("某资产报价缺口超过单页安全范围，需独立补齐历史")
        # A suspended asset's observed watermark may precede the benchmark's.
        # Include its own historical anchors rather than demanding a fake bar
        # on the benchmark's latest completed date.
        start = min(rows[-22:][0][0] for rows in observed_rows.values())
        pairs = _fetch_all(start, signal_date, fetch_pair or _fetch_pair)
        observed_clock = reference_clock + timedelta(seconds=time.monotonic() - started)
        view, updated = _build(seed, state, pairs, normalized_quotes, signal_date,
                               split_overrides, validation_time=observed_clock)
        finished_clock = reference_clock + timedelta(seconds=time.monotonic() - started)
        _quotes(quotes, signal_date, finished_clock, observe=observe)
        for code, receipt in view["metadata"]["receipts"].items():
            if receipt["current_qfq_open_anchor_provisional"]:
                _current_anchor_times(code, pairs[code], normalized_quotes[code], signal_date, finished_clock, observe=observe)
        view["metadata"]["constructed_at"] = finished_clock.isoformat()
        # The current provisional row/action remains solely in the returned view.
        _write_state(state_path, updated)
        return view
