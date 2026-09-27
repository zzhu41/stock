"""Pure, search-aware diagnostics; no files, strategy runs or model selection.

Returns use candidate-by-time shape (N,T), including the supplied first daily
return. Bootstrap resamples paired net log-return differences for the ENTIRE
supplied family. These are conditional known-history diagnostics, not SPA, PBO,
DSR, independent validation, or posterior probabilities of future performance.
"""
from datetime import datetime
import hashlib
import math

import numpy as np


DEFAULT_SEED = 20260927
MIN_DRAWS = 1000
ANNUALIZATION_DAYS = 244
LIMITATIONS = [
    "All 2014-2026 history has previously been examined; no clean out-of-sample interpretation.",
    "Conditional on the complete supplied family, not all earlier searches or the adaptive construction of this family.",
    "Unstudentized least-favorable zero-centering; this is White-style, not Hansen SPA, PBO or deflated Sharpe.",
    "Block resampling assumes the historical dependence/regime distribution is informative; it does not create new crises.",
    "An omnibus family rejection would not by itself validate the particular frozen finalist or establish causality.",
    "This test evaluates one declared clock/cost return matrix, not every drawdown, lag or multi-criterion selection requirement jointly.",
    "A p-value and Monte Carlo error are not probabilities of future profit or of overfitting.",
]


def _integer(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError("%s must be an integer >= %d" % (label, minimum))
    return int(value)


def _simple(values, label, ndim):
    out = np.asarray(values, dtype=np.float64)
    if out.ndim != ndim or not out.size or any(n == 0 for n in out.shape):
        raise ValueError(label + " has an invalid or empty shape")
    if not np.all(np.isfinite(out)) or np.any(out <= -1):
        raise ValueError(label + " must contain finite net simple returns strictly greater than -1")
    return out


def _family(returns, benchmark, candidate_ids, expected_candidate_count):
    values = _simple(returns, "Candidate (N,T) returns", 2)
    reference = _simple(benchmark, "Benchmark (T,) returns", 1)
    n, t = values.shape
    if t < 2 or len(reference) != t:
        raise ValueError("Candidate/benchmark time axes must agree and contain at least two observations")
    ids = ["candidate_%08d" % i for i in range(n)] if candidate_ids is None else list(candidate_ids)
    if len(ids) != n or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != n:
        raise ValueError("One unique nonempty candidate ID is required per row")
    if expected_candidate_count is not None and _integer(expected_candidate_count, "expected_candidate_count") != n:
        raise ValueError("Input omits or adds registered family rows; a TopK shortlist is not a full-family test")
    return values, reference, ids


def _resample_indices(t, block_length, method, rng):
    """One exact-length circular or geometric-block stationary sample."""
    if method == "circular":
        count = (t + block_length - 1) // block_length
        starts = rng.integers(0, t, size=count)
        indices = ((starts[:, None] + np.arange(block_length)) % t).ravel()[:t]
        return indices, count
    indices = np.empty(t, dtype=np.int64)
    used = blocks = 0
    while used < t:
        length = min(int(rng.geometric(1. / block_length)), t - used)
        start = int(rng.integers(0, t))
        indices[used:used + length] = (start + np.arange(length)) % t
        used += length
        blocks += 1
    return indices, blocks


def _wilson(exceedances, draws):
    """Binomial interval for simulation tail probability, not economic alpha."""
    z = 1.959963984540054
    fraction = exceedances / draws
    denominator = 1 + z * z / draws
    center = (fraction + z * z / (2 * draws)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / draws + z * z / (4 * draws * draws)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def white_style_test(returns, benchmark, block_length=20, draws=MIN_DRAWS,
                     seed=DEFAULT_SEED, method="stationary", candidate_ids=None,
                     batch_size=32, expected_candidate_count=None, return_bootstrap=False):
    """Full-family maximum mean log-excess test, with paired time resampling.

    Stationary blocks have geometric lengths with the requested mean; circular
    blocks have fixed length and the final block is truncated to exactly T.
    All rows use each draw's SAME time indices. Centering every candidate at
    zero is the least-favorable null, including poor alternatives. The winner
    field describes the observed family statistic, not a new strategy choice.

    Memory is O(N*T + batch*T + N*batch + draws), never O(N*draws*T).
    Sampling is independent of candidate count and batch size. Inputs, including
    read-only/memory-mapped arrays, are never changed. Caller must supply the
    entire registered development matrix, not only finalists or successful runs.
    """
    values, reference, ids = _family(returns, benchmark, candidate_ids, expected_candidate_count)
    block_length = _integer(block_length, "block_length")
    draws = _integer(draws, "draws", MIN_DRAWS)
    seed = _integer(seed, "seed", 0)
    batch_size = _integer(batch_size, "batch_size")
    if method not in ("stationary", "circular"):
        raise ValueError("method must be stationary or circular")
    n, t = values.shape
    if block_length > t:
        raise ValueError("block_length cannot exceed the number of observations")
    differences = np.log1p(values)  # Own allocation; never mutate supplied values.
    differences -= np.log1p(reference)[None, :]
    means = differences.mean(axis=1)
    maximum = float(means.max())
    winner = min((int(i) for i in np.flatnonzero(means == maximum)), key=lambda i: ids[i])
    observed = math.sqrt(t) * max(0., maximum)
    differences -= means[:, None]
    bootstrap = np.empty(draws, dtype=np.float64)
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    block_count = 0
    for first in range(0, draws, batch_size):
        width = min(batch_size, draws - first)
        weights = np.empty((width, t), dtype=np.float64)
        for j in range(width):
            indices, count = _resample_indices(t, block_length, method, rng)
            counts = np.bincount(indices, minlength=t)
            if int(counts.sum()) != t:
                raise AssertionError("Bootstrap dropped or added observations")
            digest.update(counts.astype("<i8", copy=False).tobytes())
            weights[j] = counts
            block_count += count
        # The only candidate-by-draw temporary is N by this small batch.
        statistics = differences @ weights.T / math.sqrt(t)
        bootstrap[first:first + width] = np.maximum(0., statistics.max(axis=0))
    exceedances = int(np.count_nonzero(bootstrap >= observed))
    p = (exceedances + 1.) / (draws + 1.)
    out = dict(method="white_style_zero_centered_%s_block_max_mean_log_excess" % method,
               return_matrix_orientation="candidate_by_time", candidates=n, trials=n,
               observations=t, draws=draws, seed=seed, batch_size=batch_size,
               block_method=method, block_length=block_length,
               circular_full_blocks=t // block_length if method == "circular" else None,
               circular_remainder=t % block_length if method == "circular" else None,
               mean_generated_block_count=block_count / draws,
               resampling_count_sha256=digest.hexdigest(),
               observed_statistic=float(observed), observed_winner_id=ids[winner],
               observed_winner_index=winner, observed_winner_mean_daily_log_excess=maximum,
               observed_winner_annual_log_excess=maximum * ANNUALIZATION_DAYS,
               annual_log_excess_is_cagr_difference=False,
               bootstrap_95_critical=float(np.quantile(bootstrap, .95)),
               bootstrap_exceedances=exceedances, p_value=float(p),
               mc_standard_error=math.sqrt(p * (1 - p) / (draws + 1)),
               bootstrap_tail_probability_wilson95=_wilson(exceedances, draws),
               minimum_reportable_p=1. / (draws + 1),
               mc_interval_note="Simulation tail-probability interval only; boundary MC SE=0 is not certainty about the true tail probability.",
               null_hypothesis="Every supplied candidate has expected net daily log-return excess <= 0 versus the benchmark",
               supplied_family_count_checked=expected_candidate_count is not None,
               scope="All supplied rows, no internal performance filter; completeness beyond the declared row count remains caller responsibility",
               candidate_selection_performed=False, clean_oos=False, limitations=list(LIMITATIONS))
    if return_bootstrap:
        out["bootstrap_statistics"] = bootstrap.tolist()
    return out


def _dates(dates, length):
    if dates is None:
        return None
    values = list(dates)
    if len(values) != length or any(not isinstance(day, str) for day in values):
        raise ValueError("One ISO date is required per return observation")
    if values != sorted(set(values)):
        raise ValueError("Dates must be unique and ascending")
    if any(datetime.strptime(day, "%Y-%m-%d").strftime("%Y-%m-%d") != day for day in values):
        raise ValueError("Dates must be canonical ISO dates")
    return values


def development_family_tests(returns, benchmark, dates, candidate_ids=None,
                             block_lengths=(20, 60), **kwargs):
    """Protocol wrapper: refuse confirmation/2026 observations before testing."""
    values = np.asarray(returns)
    if values.ndim != 2:
        raise ValueError("Candidate returns require (N,T) shape")
    days = _dates(dates, values.shape[1])
    if not days or days[0] < "2014-01-01" or days[-1] > "2021-12-31":
        raise ValueError("V11 family testing is restricted to development 2014-2021")
    lengths = list(block_lengths)
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError("Distinct registered block lengths are required")
    return dict(start=days[0], end=days[-1], clean_oos=False,
                known_confirmation_or_2026_included=False,
                tests={str(length): white_style_test(values, benchmark, block_length=length,
                                                     candidate_ids=candidate_ids, **kwargs)
                       for length in lengths})


def _paired(selected, benchmark):
    selected = _simple(selected, "Selected returns", 1)
    benchmark = _simple(benchmark, "Benchmark returns", 1)
    if selected.shape != benchmark.shape:
        raise ValueError("Selected and benchmark dates must align exactly")
    logs, baseline = np.log1p(selected), np.log1p(benchmark)
    return selected, benchmark, logs, baseline, logs - baseline


def advantage_concentration(selected, benchmark, dates=None, top_k=(1, 5, 10, 20)):
    """Positive relative-performance days, not total strategy-profit shares.

    Net-share percentages can exceed 100% when negative relative days offset
    positive days. Net-share is undefined when total net log excess <= 1e-12;
    the signed net amount is still reported. Positive-mass share is distinct.
    Removing days is accounting on a fixed path, never a tradable strategy.
    """
    a, b, logs, base, delta = _paired(selected, benchmark)
    days = _dates(dates, len(a))
    counts = [_integer(k, "top_k") for k in top_k]
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("Provide distinct positive top_k values")
    order = sorted((int(i) for i in np.flatnonzero(delta > 0)), key=lambda i: (-delta[i], i))
    net = float(delta.sum())
    positive = float(delta[delta > 0].sum())
    negative = float(delta[delta < 0].sum())
    top = {}
    for k in counts:
        indices = order[:k]
        amount = float(delta[indices].sum())
        top[str(k)] = dict(actual_positive_days=len(indices), positive_log_excess=amount,
                           share_of_net_log_excess=amount / net if net > 1e-12 else None,
                           share_of_positive_log_excess=amount / positive if positive > 0 else None,
                           fixed_path_net_log_excess_excluding_days=net - amount,
                           indices=indices, dates=[days[i] for i in indices] if days is not None else None)
    return dict(observations=len(a), selected_net_log_growth=float(logs.sum()),
                benchmark_net_log_growth=float(base.sum()), net_log_excess=net,
                positive_log_excess=positive, negative_log_excess=negative,
                positive_advantage_days=len(order), negative_advantage_days=int(np.count_nonzero(delta < 0)),
                net_share_defined=net > 1e-12, top=top,
                top_positive_days=[dict(index=i, date=days[i] if days is not None else None,
                    selected_return=float(a[i]), benchmark_return=float(b[i]), log_excess=float(delta[i]))
                    for i in order[:max(counts)]],
                interpretation="Shares of paired log excess, not shares of total strategy profit; signed offsets can make net shares exceed 100%.",
                exclusion_is_tradable=False, clean_oos=False)


def event_cluster_diagnostics(selected, benchmark, dates, event_dates,
                              merge_gap=20, pre=5, post=20):
    """Descriptive event clusters and clipped pre/core/post fixed-path windows.

    Adjacent triggers <= merge_gap market observations apart belong to one
    cluster. Core includes first through last trigger; pre/post exclude that
    core. Neighboring clusters' surrounding windows may overlap and MUST NOT
    be summed or treated as independent observations. No re-selection/rerun.
    """
    a, b, logs, base, delta = _paired(selected, benchmark)
    days = _dates(dates, len(a))
    if days is None:
        raise ValueError("Event diagnostics require an explicit date axis")
    merge_gap = _integer(merge_gap, "merge_gap", 0)
    pre, post = _integer(pre, "pre", 0), _integer(post, "post", 0)
    lookup = {day: i for i, day in enumerate(days)}
    supplied = list(event_dates)
    if any(day not in lookup for day in supplied):
        raise ValueError("Every event must be an observed date; no nearest-date reassignment")
    indices = sorted({lookup[day] for day in supplied})
    grouped = []
    for i in indices:
        if not grouped or i - grouped[-1][-1] > merge_gap:
            grouped.append([i])
        else:
            grouped[-1].append(i)

    def interval(lo, hi):
        if lo >= hi:
            return None
        return dict(start=days[lo], end=days[hi - 1], observations=hi - lo,
                    selected_log_growth=float(logs[lo:hi].sum()),
                    benchmark_log_growth=float(base[lo:hi].sum()),
                    net_log_excess=float(delta[lo:hi].sum()))

    clusters, windows = [], []
    for members in grouped:
        first, last = members[0], members[-1]
        left, right = max(0, first - pre), min(len(days), last + post + 1)
        windows.append((left, right))
        clusters.append(dict(event_dates=[days[i] for i in members], event_count=len(members),
                             first_event=days[first], last_event=days[last],
                             pre=interval(left, first), core=interval(first, last + 1),
                             post=interval(last + 1, right),
                             pre_truncated=first < pre, post_truncated=last + post >= len(days)))
    overlapping = [(i - 1, i) for i in range(1, len(windows)) if windows[i][0] < windows[i - 1][1]]
    return dict(supplied_event_count=len(supplied), unique_event_dates=len(indices),
                cluster_count=len(clusters), merge_gap_observations=merge_gap,
                pre_observations=pre, post_observations=post, clusters=clusters,
                overlapping_surrounding_window_pairs=overlapping,
                causal_event_effect=False, independent_event_samples=False,
                interpretation="Fixed-path conditional description; trigger-day market returns may belong to the pre-trade holding. Overlapping windows are not independent and must not be added.",
                clean_oos=False)
