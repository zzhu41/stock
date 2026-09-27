"""Export comparable corrected-close NAV and drawdown figures."""
from datetime import datetime
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from .data import BASE


def main():
    out = BASE / "corrected_results"
    paths = json.loads((out / "selected_paths.json").read_text())
    selection = json.loads((out / "selection.json").read_text())
    dates = [datetime.strptime(x, "%Y-%m-%d") for x in json.loads((out / "full_dates.json").read_text())]
    primary, guarded = selection["primary_return_champion"], selection["risk_guarded_candidate"]
    curves = [("c_v9", "v9", "#9ba6b1"), ("c_v91", "v9.1", "#6c82a0"),
              ("c_v92", "v9.2", "#284b75"), (primary, "New H / return priority", "#cb4b36")]
    if guarded and guarded != primary:
        curves.append((guarded, "New H / risk gates", "#138780"))
    fig, (ax, dd) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name, label, color in curves:
        nav = np.asarray(paths[name]["navs"])
        ax.plot(dates, nav, label=label, color=color, linewidth=1.6)
        drawdown = nav / np.maximum.accumulate(np.concatenate(([1.0], nav)))[1:] - 1
        dd.plot(dates, drawdown, color=color, linewidth=1.1)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1 (log scale)")
    dd.set_ylabel("Drawdown")
    dd.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper left", frameon=False)
    for panel in (ax, dd):
        panel.grid(True, alpha=.15)
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)
        panel.axvspan(datetime(2026, 1, 1), dates[-1], color="#edce93", alpha=.25)
    fig.suptitle("Corrected-close research comparison | 2014-01-02 to 2026-09-24", fontsize=14, x=.08, ha="left")
    fig.text(.08, .935, "Same-close fills, 1 bp per side; instant free dividend reinvestment assumed. Already-researched history.", fontsize=9, color="#555555")
    fig.text(.08, .02, "Shaded: 2026 report-only period (not clean OOS). Original additive-QFQ results excluded. No live execution claim.", fontsize=9, color="#555555")
    fig.subplots_adjust(top=.90, bottom=.10, left=.08, right=.98, hspace=.08)
    fig.savefig(out / "comparison.png", dpi=160)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
