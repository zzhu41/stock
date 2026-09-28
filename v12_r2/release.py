"""Export year tables and seal new R2 artifacts without modifying old work."""
import argparse
import csv
import json
from pathlib import Path

from .data import BASE,sha,dump,verify_inputs
from .research import load_results,stamp


def verify():
    verify_inputs();load_results()
    receipt=json.loads((BASE/'release_receipt.json').read_text())
    for name,expected in receipt['file_sha256'].items():
        if sha(BASE/name)!=expected:raise ValueError('R2 release artifact changed: '+name)
    return dict(passed=True,files=len(receipt['file_sha256']),primary=receipt['primary'],balanced_reference=receipt['balanced_reference'],deployed=False)


def seal():
    if (BASE/'release_receipt.json').exists():return verify()
    verify_inputs();registry,evaluation,selection=load_results()
    completion=json.loads((BASE/'completion.json').read_text())
    for name,expected in completion['file_sha256'].items():
        if sha(BASE/name)!=expected:raise ValueError('Completed evidence changed: '+name)
    post=json.loads((BASE/'results/post_registration.json').read_text())['definition']
    roles=post['roles'];rows={r['id']:r for r in evaluation['rows']}
    with (BASE/'results/yearly.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['role','id','scenario','year','total_return','max_dd','sessions'])
        for role,cid in roles.items():
            for scenario,s in rows[cid]['scenarios'].items():
                for year,m in s['yearly'].items():w.writerow([role,cid,scenario,year,m['total_return'],m['max_dd'],m['sessions']])
    local=json.loads((BASE/'results/local_diagnostics.json').read_text())
    with (BASE/'results/local_sensitivity.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['parent_id','kind','axis','label','id','cagr_delta_pp','return_2021_delta_pp','return_2026_delta_pp'])
        for cid,data in local.items():
            for kind in ('neighbors','restore_one_C4_axis'):
                for row in data[kind]:
                    delta=row['delta'];w.writerow([cid,kind,row['axis'],row.get('label','C4'),row['id'],delta['cagr_pp'],delta['return_2021_pp'],delta['return_2026_pp']])
    labels={'c4':'C4','c':'C对照','h':'H','simple':'V9.2+','highest_cagr':'最高年化（未合格）',
            'highest_2021':'最高2021（未合格）','main_three_gates_1':'三主指标达标（压力失败）'}
    lines=['# R2逐年和局部表','','全部是已知历史结果，不是新的实盘策略。严格和折衷资格均为0。',
           '“三主指标达标”仅指全期年化不低C4、回撤≤21%、2021至少恢复C；该唯一配置的2026及延迟压力仍失败，不是另设的晋级层或推荐。','',
           '| 年份 | '+' | '.join(labels.get(role,role) for role in roles)+' |','|---|'+'---:|'*len(roles)]
    for year in sorted(rows[next(iter(roles.values()))]['scenarios']['close_1bp']['yearly']):
        lines.append('| '+year+' | '+' | '.join('%.3f%%'%(rows[cid]['scenarios']['close_1bp']['yearly'][year]['total_return']*100) for cid in roles.values())+' |')
    lines+=['','2026截至09-24，非全年。所有年份包含年界首日的真实旧持仓收益，没有免费年度重启。','',
            '## 局部敏感性（全部点已在主网格，无新增回测）','','| 解释项ID | 邻居数 | 邻居自身年化差中位数 / Q25(pp) |','|---|---:|---:|']
    for cid,x in local.items():lines.append('| %s | %d | %.3f / %.3f |'%(cid,len(x['neighbors']),x['neighbor_cagr_delta_median_pp'],x['neighbor_cagr_delta_q25_pp']))
    lines+=['','[四压力年表CSV](results/yearly.csv) · [所有单轴变化CSV](results/local_sensitivity.csv)','',
        '![全程与2021曲线](results/comparison.png)','',
        '解释项可只读查询：`python3.8 -B -m v12_r2.inspect highest_cagr` 或 `main_three_gates_1`。输出会保留未过门槛，不冒名为主候选。','']
    (BASE/'RESULTS.md').write_text('\n'.join(lines))
    (BASE/'REPRODUCE.md').write_text('# 只读复核\n\n```bash\npython3.8 -B -m v12_r2.release\npython3.8 -B -m v12_r2.cli backtest c4\npython3.8 -B -m v12_r2.inspect main_three_gates_1 --scenario lag1_11bp\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3.8 -B -m unittest discover -s v12_r2/tests -v\n```\n\n主扫描、选择、统计均已冻结；入口拒绝覆盖选择。`features.npz`、完整866条四情景`paths.npz`和独立参考矩阵保留在本目录，不需要重建旧cache。矩阵和JSON均小于100MB。\n\n只校验明确的冻结模型/行情/证据；不会把另外获准维护的生产推送代码误判为研究修改。没有网络访问、真实账户写入或订单。\n')
    files={str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.is_file() and 'cache' not in p.parts and '__pycache__' not in p.parts and p.name!='release_receipt.json'}
    receipt=dict(sealed_at=stamp(),file_sha256=files,primary=selection['primary'],balanced_reference=selection['balanced_reference'],
                 deployed=False,clean_oos=False,old_input_count=len(verify_inputs()['sha256']))
    dump(BASE/'release_receipt.json',receipt)
    return verify()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');a=p.parse_args()
    print(json.dumps(seal() if a.seal else verify(),ensure_ascii=False,indent=2))
