"""Shared dated indicator bank; immutable between search candidates.

Only ranking changes across this registry. Entry/exit/buffer/crash switches
affect decisions, not these price features. No future observation is sliced in.
"""
import bisect
import json

from v10_next.data import configure, wls_score
from v10_next.frozen import strategy
from .data import ASSET_ORDER, BASE, START, CASH


DYNAMIC = ("mom5", "mom20", "mom60", "vol", "score", "above_ma", "ret1", "mom20_max",
           "pos_frac", "max_ret", "dvol", "ma_rising", "dist_ma250")
CONSTANTS = dict(dif=0.0, dea=0.0, j_val=0.0, rsi=0.0, ma_align=True, k_val=50.0, d_val=50.0,
                 above_ma20=True, ma20_rising=True, bias20=0.0, pctb=0.0, cci=0.0,
                 hist_pct=0.0, donchian_low=0.0, above_donchian=True, er20=1.0,
                 rets_tail=None, vol_ratio=0.0, vol_in_ratio=1.0, mom10_z=0.0, slope_pos_days=0)


class FeatureBank:
    def __init__(self, histories, calendar):
        self.histories, self.calendar = histories, tuple(calendar)
        self.dates = {c: [r[0] for r in rows] for c, rows in histories.items()}
        self.closes = {c: [r[2] for r in rows] for c, rows in histories.items()}
        self.volumes = {c: [r[3] for r in rows] for c, rows in histories.items()}
        self.presets = json.loads((BASE.parent / "v10_next/frozen/presets.json").read_text())
        configure(self.presets["v9.1"])
        self.base = []
        self.row_indices = []
        self.rank_cache = {}
        self.window_scores = {}
        for date in self.calendar:
            row, indices = {}, {}
            if date >= START:
                for code in ASSET_ORDER:
                    if code == CASH or code not in histories:
                        continue
                    ds = self.dates[code]
                    j = bisect.bisect_right(ds, date) - 1
                    if j < 269 or ds[j] != date:
                        continue
                    ind = strategy.indicators(self.closes[code][:j + 1], self.volumes[code][:j + 1])
                    if ind is None:
                        continue
                    small = {k: ind[k] for k in DYNAMIC}
                    avg = sum(self.volumes[code][j - 20:j]) / 20
                    small["volume_ratio20"] = self.volumes[code][j] / avg if avg > 0 else 0.0
                    row[code], indices[code] = small, j
            self.base.append(row)
            self.row_indices.append(indices)

    def _scores(self, window):
        if window not in self.window_scores:
            rows = []
            for i, base in enumerate(self.base):
                row = {}
                for code, ind in base.items():
                    if window == 25:
                        row[code] = ind["score"]
                    else:
                        j = self.row_indices[i][code]
                        row[code] = wls_score(self.closes[code][:j + 1], window, ind["vol"])
                rows.append(row)
            self.window_scores[window] = rows
        return self.window_scores[window]

    def ranked(self, mode, windows):
        key = (mode, tuple(windows))
        if key not in self.rank_cache:
            if mode == "wls":
                if not windows:
                    raise ValueError("WLS needs at least one window")
                scores = [self._scores(w) for w in windows]
            elif mode not in ("mom20_vol", "mom60_vol", "mom20"):
                raise ValueError("Unsupported score mode")
            result = []
            for i, base in enumerate(self.base):
                row = []
                for code, ind in base.items():
                    if mode == "wls":
                        score = sum(s[i][code] for s in scores) / len(scores)
                    elif mode == "mom20":
                        score = ind["mom20"]
                    else:
                        score = ind["mom60" if mode == "mom60_vol" else "mom20"] / ind["vol"] if ind["vol"] else 0.0
                    small = dict(ind, score=score)
                    row.append((code, small))
                row.sort(key=lambda x: x[1]["score"], reverse=True)
                result.append(row)
            self.rank_cache[key] = result
        return self.rank_cache[key]

    def table(self, i, mode, windows, codes, oracle=False):
        rows = [(c, ind) for c, ind in self.ranked(mode, windows)[i] if c in codes]
        if oracle:
            return [(c, dict(CONSTANTS, **ind, mom20_peak20=ind["mom20_max"])) for c, ind in rows]
        return rows
