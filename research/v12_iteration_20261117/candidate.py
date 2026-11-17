"""Read-only research CLI for registered alternatives; never generates live orders.

Example: python3.8 -B research/v12_iteration_20261117/candidate.py \
    --variant defensive_lowvol --start 2026-09-29 --fee-bp 11
"""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from study import CANDIDATES, summarize
from data_bundle import load_dataset
from engine import run_portfolio

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=[n for n,_,_ in CANDIDATES])
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--start',default='2014-01-02')
    parser.add_argument('--end')
    parser.add_argument('--fee-bp',type=float,default=1.)
    parser.add_argument('--lag',type=int,choices=[0,1],default=0)
    args=parser.parse_args()
    if args.list:
        print(json.dumps([dict(name=n,spec=s,selectable=q) for n,s,q in CANDIDATES],ensure_ascii=False,indent=2));return
    if args.variant is None: parser.error('--variant is required; no alternative has qualified as an upgrade')
    spec=next(s for n,s,_ in CANDIDATES if n==args.variant)
    arrays,meta,cfg=load_dataset()
    result=run_portfolio(arrays,meta,cfg,spec,start=args.start,end=args.end,
                         fee=args.fee_bp/10000,lag=args.lag)
    last=result['trace'][-1]
    output=dict(variant=args.variant,spec=spec,period=[result['dates'][0],result['dates'][-1]],
        scenario=dict(lag=args.lag,one_way_fee_bp=args.fee_bp),metrics=summarize(result),
        final_hypothetical_holdings=last['weights'],
        frozen_completed_data=meta['dates'][-1],order_submission=False,production_integration=False,
        qualified_upgrade=False,
        qualification_note='None of the alternatives passed the preregistered upgrade gates. This is a historical research replay.')
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
