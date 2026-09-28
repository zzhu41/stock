"""Read-only explanatory-role inspection; every failed gate stays visible."""
import argparse
import json
from .data import BASE,verify_inputs,sha
from .research import load_results


def inspect_role(role):
    verify_inputs();registry,evaluation,selection=load_results()
    post=json.loads((BASE/'results/post_registration.json').read_text())['definition']
    if post['source_sha256']!=sha(BASE/'post.py') or post['frozen_selection_sha256']!=sha(BASE/'results/selection.json'):
        raise ValueError('Frozen explanatory analysis changed')
    if role not in post['roles']:raise ValueError('Unknown explicit explanatory role')
    cid=post['roles'][role];record=next(x for x in registry['records'] if x['id']==cid)
    row=next(x for x in evaluation['rows'] if x['id']==cid)
    return dict(role=role,id=cid,record=record,scenarios=row['scenarios'],
        strict_qualified=cid in selection['qualified_ids']['strict'],balanced_qualified=cid in selection['qualified_ids']['balanced'],
        failed_gates={tier:[k for k,v in selection['checks'][cid][tier].items() if not v] for tier in ('strict','balanced')},
        status='Explicit explanatory historical result; not a substitute qualified strategy',clean_oos=False,deployed=False,order_submission=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('role');p.add_argument('--scenario',default='close_1bp',choices=('close_1bp','close_11bp','lag1_1bp','lag1_11bp'));a=p.parse_args()
    value=inspect_role(a.role);value['scenario']=a.scenario;value['metrics']=value.pop('scenarios')[a.scenario]
    print(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
