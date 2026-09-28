"""Explicit read-only access to three sealed, unqualified R2 explanations.

Only stored continuous paths are sliced. No model evaluation, selection,
network calls, account updates, or orders occur in this module.
"""
import argparse
from bisect import bisect_right
from copy import deepcopy
import json

import numpy as np

from .data import BASE,END,sha,verify_inputs
from .research import load_results
from .features import effective_config
from v10_deep.scan import metrics

FROZEN_RELEASE_SHA256='f05425fd1e4f599f780d89d785ce93074e1d3af2fe7ea4c8832ad7e9058582eb'
FIXED_IDS={
    'highest_cagr':'v12r2_08b522c2e1aa291f054d',
    'highest_2021':'v12r2_7743224871bb70b78781',
    'main_three_gates_1':'v12r2_9a8799a5f7dd7581608e',
}
ALIASES={name:name for name in FIXED_IDS}
ALIASES['recovery_2021']='main_three_gates_1'


def _frozen():
    verify_inputs()
    registry,evaluation,selection=load_results()
    release_path=BASE/'release_receipt.json'
    if sha(release_path)!=FROZEN_RELEASE_SHA256:
        raise ValueError('The original R2 release receipt changed')
    receipt=json.loads(release_path.read_text())
    for name,expected in receipt['file_sha256'].items():
        if sha(BASE/name)!=expected:
            raise ValueError('A captured R2 release artifact changed: '+name)
    post=json.loads((BASE/'results/post_registration.json').read_text())['definition']
    if selection['primary'] is not None or selection['balanced_reference'] is not None:
        raise ValueError('Frozen unqualified selection status changed')
    if (selection['exploratory']['highest_cagr']!=FIXED_IDS['highest_cagr']
            or selection['exploratory']['highest_2021']!=FIXED_IDS['highest_2021']
            or post['roles']['main_three_gates_1']!=FIXED_IDS['main_three_gates_1']):
        raise ValueError('The three fixed explanatory identities changed')
    return registry,evaluation,selection


def evaluate(role, end=END, lag=0, fee_bp=1):
    """Read one explicit role; an arbitrary candidate ID is never accepted."""
    if role not in ALIASES:
        raise ValueError('An explicit fixed explanatory role is required; arbitrary IDs and primary substitutes are forbidden')
    if type(lag) is not int or lag not in (0,1) or type(fee_bp) is not int or fee_bp not in (1,11):
        raise ValueError('Only lag0/lag1 times 1bp/11bp are registered')
    registry,unused,selection=_frozen()
    source_role=ALIASES[role]
    cid=FIXED_IDS[source_role]
    metadata=json.loads((BASE/'results/path_metadata.json').read_text())
    if end not in metadata['dates']:
        raise ValueError('End must be an observed date within the frozen 2014–2026-09-24 path')
    stop=bisect_right(metadata['dates'],end)
    dates=metadata['dates'][:stop]
    scenario=('lag1' if lag else 'close')+'_%dbp'%fee_bp
    index=metadata['ids'].index(cid)
    with np.load(BASE/'results/paths.npz',allow_pickle=False) as stored:
        returns=stored[scenario+'__returns'][index,:stop]
        holdings=stored[scenario+'__holdings'][index,:stop]
    record=next(r for r in registry['records'] if r['id']==cid)
    assets=json.loads((BASE/'results/feature_metadata.json').read_text())['assets']
    years=sorted({d[:4] for d in dates})
    yearly={year:metrics(returns[np.asarray([d[:4]==year for d in dates])]) for year in years}
    return dict(requested_role=role,source_role=source_role,candidate_id=cid,
        status='unqualified diagnostic only; not deployed',strict_qualified=False,balanced_qualified=False,
        calculation_mode='Read stored continuous paths and recompute slice metrics only; the strategy is not rerun',
        official_primary=None,official_balanced_reference=None,clean_oos=False,deployed=False,order_submission=False,
        feature_spec=deepcopy(record['feature_spec']),effective_config=effective_config(record,lag),
        scenario=scenario,execution=dict(lag_closes=lag,fee_per_leg_bp=fee_bp,first_session_free=True,
            annualization_sessions=244,price_basis='corrected TR, ideal close; open is not used'),
        period=[dates[0],dates[-1]],full=metrics(returns),yearly=yearly,
        switches=int(np.count_nonzero(holdings[1:]!=holdings[:-1])),
        historical_close_target=assets[int(holdings[-1])] if holdings[-1]>=0 else None,
        full_frozen_period_failed_gates={tier:[k for k,v in selection['checks'][cid][tier].items() if not v]
                                        for tier in ('strict','balanced')},
        qualification_note='Qualification remains the original full-period verdict; a shorter slice never rescues or reselects a model',
        slicing_note='Original continuous returns from 2014 are sliced at end; each calendar-year slice includes its first-day return, without a free restart',
        annual_return_note='yearly.total_return is calendar-period cumulative return; the last year may be incomplete',
        signal_note='historical_close_target is a frozen historical model holding, not a current order instruction',
        provenance=dict(original_release_sha256=FROZEN_RELEASE_SHA256,
            selection_sha256=sha(BASE/'results/selection.json'),path_sha256=selection['paths_sha256'],
            role_source='selection.exploratory' if source_role!='main_three_gates_1' else 'sealed post_registration.roles'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role',choices=tuple(ALIASES),help='Required fixed explanatory role; no default winner')
    parser.add_argument('--end',default=END)
    parser.add_argument('--lag',type=int,choices=(0,1),default=0)
    parser.add_argument('--fee-bp',type=int,choices=(1,11),default=1)
    args=parser.parse_args()
    print(json.dumps(evaluate(args.role,args.end,args.lag,args.fee_bp),ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
