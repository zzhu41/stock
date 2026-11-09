"""Read frozen R2 evidence only; print independent metrics and provenance.

Run: python3.8 -B research/rsi_20261109/verify_evidence.py
Does not import strategies, fetch prices, select candidates, or write accounts.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'v12_r2'
CID = 'v12r2_08b522c2e1aa291f054d'
RELEASE_SHA = 'f05425fd1e4f599f780d89d785ce93074e1d3af2fe7ea4c8832ad7e9058582eb'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sha(BASE / 'release_receipt.json') != RELEASE_SHA:
        raise ValueError('Frozen release receipt changed')
    receipt = json.loads((BASE / 'release_receipt.json').read_text())
    for name, expected in receipt['file_sha256'].items():
        if sha(BASE / name) != expected:
            raise ValueError('Release artifact changed: ' + name)
    frozen = json.loads((BASE / 'frozen_inputs.json').read_text())
    for name, expected in frozen['sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen input changed: ' + name)
    meta = json.loads((BASE / 'results/path_metadata.json').read_text())
    selection = json.loads((BASE / 'results/selection.json').read_text())
    registry = json.loads((BASE / 'registered_candidates.json').read_text())
    index = meta['ids'].index(CID)
    scenarios = {}
    with np.load(BASE / 'results/paths.npz', allow_pickle=False) as paths:
        for name in ('close_1bp', 'close_11bp', 'lag1_1bp', 'lag1_11bp'):
            returns = paths[name + '__returns'][index]
            holdings = paths[name + '__holdings'][index]
            if (returns.ndim != 1 or holdings.ndim != 1
                    or len(returns) != len(meta['dates'])
                    or len(holdings) != len(returns)
                    or not np.all(np.isfinite(returns))
                    or np.any(returns <= -1)):
                raise ValueError('Invalid frozen path: ' + name)
            nav = np.cumprod(1 + returns)
            peaks = np.maximum.accumulate(np.r_[1., nav])[1:]
            scenarios[name] = dict(
                cagr=float(np.expm1(np.log1p(returns).sum() * 244 / len(returns))),
                max_drawdown=float(np.min(nav / peaks - 1)),
                final_nav=float(nav[-1]),
                switches=int(np.count_nonzero(holdings[1:] != holdings[:-1])))
    result = dict(
        research_date='2026-11-09', candidate_id=CID,
        period=[meta['dates'][0], meta['dates'][-1]],
        observations=len(meta['dates']), annualization_sessions=244,
        records=len(meta['ids']),
        selectable_records=sum(r['selectable'] for r in registry['records']),
        strict_selection=selection['primary'],
        balanced_selection=selection['balanced_reference'],
        release_files_verified=len(receipt['file_sha256']),
        frozen_inputs_verified=len(frozen['sha256']), scenarios=scenarios,
        artifact_sha256={name: sha(BASE / name) for name in (
            'release_receipt.json', 'results/paths.npz', 'results/selection.json')},
        scope='Independent recomputation of stored historical paths, not a new strategy backtest.',
        limitations=[
            'All historical observations were exposed to earlier research.',
            'lag1 means next close, not next open or the live 14:50 clock.',
            'Historical research qualification and later user-directed deployment are distinct.',
            'No new candidate was searched or promoted by this script.'])
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
