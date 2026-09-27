"""Search-aware diagnostics for a supplied complete, already researched family.

The max test is an unstudentized, zero-centered circular-block White-style
diagnostic of mean net log-return differences. It is not a full Hansen SPA,
PBO, a claim about all previous searches, or a future profit probability.
Walk-forward output selects supplied model return streams; it does not rebuild
portfolio transfers at model-switch boundaries and is not a tradable backtest.
"""
import numpy as np


TRADING_DAYS = 244
WALKFORWARD_FOLDS = (
    ("2014-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),
    ("2014-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
    ("2014-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),
    ("2014-01-01", "2023-12-31", "2024-01-01", "2025-12-31"),
)


def _inputs(returns, benchmark_returns, candidate_ids):
    values = np.asarray(returns, dtype=np.float64)
    benchmark = np.asarray(benchmark_returns, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("Candidate returns must be a nonempty T by M matrix")
    if benchmark.ndim != 1 or len(benchmark) != values.shape[0]:
        raise ValueError("Benchmark must have exactly the same T observations")
    if (not np.all(np.isfinite(values)) or not np.all(np.isfinite(benchmark))
            or np.any(values <= -1) or np.any(benchmark <= -1)):
        raise ValueError("Net simple returns must be finite and strictly greater than -1")
    ids = (["candidate_%d" % i for i in range(values.shape[1])]
           if candidate_ids is None else list(candidate_ids))
    if len(ids) != values.shape[1] or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("Supply one nonempty string ID per candidate")
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be unique")
    return values, benchmark, ids


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _best_index(means, ids):
    maximum = np.max(means)
    tied = np.flatnonzero(means == maximum)
    return min((int(i) for i in tied), key=lambda i: ids[i])


def _annualized(logs):
    return float(np.expm1(np.mean(logs) * TRADING_DAYS))


def white_style_max_test(returns, benchmark_returns, block_size=60, draws=1000,
                         seed=20260927, candidate_ids=None, batch_size=32):
    """Test the entire supplied family's maximum mean net log outperformance.

    Every bootstrap draw uses the same circular block indices for all M columns
    (and hence their paired benchmark differences). The final block is truncated
    to retain exactly T returns, including the supplied initial-entry return.
    All columns, including poor alternatives, enter the maximum; there is no
    performance-based shortlist inside this function.
    """
    values, benchmark, ids = _inputs(returns, benchmark_returns, candidate_ids)
    block_size = _positive_integer(block_size, "block_size")
    draws = _positive_integer(draws, "draws")
    batch_size = _positive_integer(batch_size, "batch_size")
    n, m = values.shape
    if block_size > n:
        raise ValueError("block_size cannot exceed the number of observations")

    differences = np.log1p(values) - np.log1p(benchmark)[:, None]
    means = differences.mean(axis=0)
    best = _best_index(means, ids)
    observed = float(np.sqrt(n) * max(0.0, means[best]))
    differences -= means  # least-favorable zero-centering of every candidate

    # Circular rolling sums. Blocks shorter than T preserve local dependence;
    # concatenating their sums never re-executes a strategy on synthetic prices.
    padded = np.concatenate((differences, differences[:block_size]), axis=0)
    prefix = np.concatenate((np.zeros((1, m)), np.cumsum(padded, axis=0)), axis=0)
    rolling = prefix[block_size:block_size + n] - prefix[:n]
    full, remainder = divmod(n, block_size)
    blocks = full + bool(remainder)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=np.float64)
    for first in range(0, draws, batch_size):
        width = min(batch_size, draws - first)
        # Row-major draws make the random sequence independent of batch_size.
        starts = rng.integers(0, n, size=(width, blocks))
        totals = rolling[starts[:, :full]].sum(axis=1)
        if remainder:
            tail = starts[:, -1]
            totals += prefix[tail + remainder] - prefix[tail]
        bootstrap[first:first + width] = np.maximum(0.0, totals.max(axis=1) / np.sqrt(n))

    exceedances = int(np.count_nonzero(bootstrap >= observed))
    p_value = (1.0 + exceedances) / (draws + 1.0)
    return dict(
        method="white_style_zero_centered_circular_block_max_mean_log_difference",
        observations=n, tested_candidate_count=m, block_size=block_size,
        full_blocks=full, remainder_observations=remainder, draws=draws,
        seed=int(seed), batch_size=batch_size,
        null_hypothesis="Every supplied candidate has expected net log-return difference <= 0 versus the benchmark",
        observed_statistic=observed, best_candidate_id=ids[best], best_candidate_index=best,
        best_mean_daily_log_difference=float(means[best]),
        bootstrap_95_critical=float(np.quantile(bootstrap, .95)),
        bootstrap_exceedances=exceedances, p_value=float(p_value),
        p_value_monte_carlo_se=float(np.sqrt(p_value * (1 - p_value) / (draws + 1))),
        scope="All supplied columns; caller must supply the entire preregistered candidate family, not TopK",
        limitations=[
            "Previously examined historical data; no clean OOS interpretation.",
            "Conditional on this fixed supplied search family, not every earlier or unrecorded search.",
            "Assumes the historical dependence and regime distribution support block resampling.",
            "Unstudentized least-favorable zero-centering; not Hansen SPA and can be conservative with poor alternatives.",
            "A p-value is not PBO or a posterior probability of future outperformance.",
        ],
    )


def walkforward_selection_diagnostic(returns, benchmark_returns, dates, candidate_ids=None):
    """Four frozen expanding-training / later-block return-stream selections.

    In each training block choose highest net log growth, tie by candidate ID,
    but select the benchmark if that candidate does not strictly outperform it.
    This selection rule deliberately differs from the main champion's two-part
    development eligibility gate. The matrix contains precomputed model account
    returns: stitched output omits strategy-switch transfers and related costs.
    """
    values, benchmark, ids = _inputs(returns, benchmark_returns, candidate_ids)
    dates = list(dates)
    if len(dates) != values.shape[0] or any(not isinstance(d, str) for d in dates):
        raise ValueError("Supply one sorted date string per return observation")
    if dates != sorted(set(dates)):
        raise ValueError("Dates must be unique and sorted")
    calendar = np.asarray(dates)
    logs, baseline = np.log1p(values), np.log1p(benchmark)
    folds, stitched, baseline_stitched, stitched_dates = [], [], [], []
    for train_start, train_end, test_start, test_end in WALKFORWARD_FOLDS:
        train = (calendar >= train_start) & (calendar <= train_end)
        test = (calendar >= test_start) & (calendar <= test_end)
        if not np.any(train) or not np.any(test):
            raise ValueError("Every registered walk-forward fold needs nonempty training and test observations")
        means = logs[train].mean(axis=0)
        best = _best_index(means, ids)
        benchmark_mean = float(baseline[train].mean())
        selected = best if means[best] > benchmark_mean else None
        chosen_test = logs[test, selected] if selected is not None else baseline[test]
        chosen_train = logs[train, selected] if selected is not None else baseline[train]
        folds.append(dict(
            train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end,
            training_observations=int(train.sum()), test_observations=int(test.sum()),
            selected_id=ids[selected] if selected is not None else None,
            selected_index=selected, selected_benchmark=selected is None,
            best_training_candidate_id=ids[best],
            selected_train_cagr=_annualized(chosen_train),
            benchmark_train_cagr=_annualized(baseline[train]),
            selected_test_cagr=_annualized(chosen_test),
            benchmark_test_cagr=_annualized(baseline[test]),
            test_cagr_difference=_annualized(chosen_test) - _annualized(baseline[test]),
        ))
        stitched.append(chosen_test)
        baseline_stitched.append(baseline[test])
        stitched_dates.extend(calendar[test].tolist())
    combined = np.concatenate(stitched)
    ref = np.concatenate(baseline_stitched)
    return dict(
        method="blocked_expanding_training_reselection_of_fixed_model_return_streams",
        tested_candidate_count=values.shape[1], folds=folds,
        stitched_start=stitched_dates[0], stitched_end=stitched_dates[-1],
        stitched_observations=len(combined), selected_stream_cagr=_annualized(combined),
        benchmark_stream_cagr=_annualized(ref),
        selected_stream_total_return=float(np.expm1(combined.sum())),
        benchmark_stream_total_return=float(np.expm1(ref.sum())),
        realizable_strategy=False, clean_oos=False, includes_2026=False,
        limitations=[
            "Diagnostic rule: train-period best above benchmark, not the main two-subperiod eligibility rule.",
            "All historical periods were previously inspected; the candidate family was designed using known history.",
            "Stitched precomputed account returns do not rebuild holdings or charge model-switch transfer costs.",
            "This evaluates selection instability, not a tradable walk-forward strategy or corrected significance.",
            "2026 observations do not enter any training or test block.",
        ],
    )
