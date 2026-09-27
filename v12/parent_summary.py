"""Post-freeze contribution summary, not another strategy selection."""
import json
import numpy as np
from .data import BASE,sha,dump
from .family_diagnostics import load_concentration
from v10_deep.scan import metrics


def summarize():
    metadata=json.loads((BASE/'results/pressure/base/path_metadata.json').read_text())
    profiles=json.loads((BASE/'profiles.json').read_text())['variants']
    cutoff=sum(day<'2026-01-01' for day in metadata['dates'])
    values=load_concentration(BASE/'results/family_diagnostics','close_1bp')
    output={}
    with np.load(BASE/'results/pressure/base/paths.npz',allow_pickle=False) as archive:
        daily=archive['close_1bp__returns']
    for role in ('h','simple','v92','balanced_a','balanced_b','balanced_c'):
        cid=profiles[role]['id'];i=metadata['ids'].index(cid);concentration=values['candidates'][cid]
        output[role]=dict(known_history_pre2026=metrics(daily[i,:cutoff]),
            excess_log_2026=concentration['years']['2026']['net_log_excess'],
            share_2026_of_net_log_excess=concentration['years']['2026']['share_of_net_log_excess'],
            concentration=concentration)
    inputs=('profiles.json','results/pressure/base/path_metadata.json','results/pressure/base/paths.npz',
            'results/family_diagnostics/concentration_close_1bp.json.gz','results/family_diagnostics/concentration.json')
    dump(BASE/'results/parent_concentration.json',dict(
        interpretation='Post-freeze description, no new selection; pre2026 is also examined history, not clean OOS. All excess shares refer to paired log returns, not strategy profit.',
        pre2026_period_end='2025-12-31',roles=output,source_sha256=sha(BASE/'parent_summary.py'),
        input_sha256={name:sha(BASE/name) for name in inputs}))


if __name__=='__main__':summarize()
