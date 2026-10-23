"""Freeze a broad, explicitly defined technology research universe."""
import json
import shutil
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
CORE = ["半导体", "元件", "消费电子", "光学光电子", "其他电子Ⅱ", "电子化学品Ⅱ",
        "通信设备", "通信服务", "软件开发", "IT服务Ⅱ", "计算机设备", "军工电子Ⅱ",
        "自动化设备", "游戏Ⅱ", "数字媒体"]
ADJACENT = ["通用设备", "专用设备", "电机Ⅱ", "其他电源设备Ⅱ", "玻璃玻纤",
            "家电零部件Ⅱ", "汽车零部件", "电网设备", "金属新材料", "化学制品",
            "塑料", "航天装备Ⅱ"]


def main():
    snapshot = pd.read_csv('/tmp/tech_screen_access/universe.csv', dtype={'code': str})
    financial = pd.read_csv('/tmp/tech_screen_financial/latest_asof_2026-10-23.csv',
                            dtype={'code': str})
    inputs = BASE / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    shutil.copy2('/tmp/tech_screen_access/universe.csv', str(inputs / 'all_a_quote_snapshot.csv'))
    shutil.copy2('/tmp/tech_screen_access/metadata.json', str(inputs / 'quote_metadata.json'))
    shutil.copy2('/tmp/tech_screen_financial/latest_asof_2026-10-23.csv',
                 str(inputs / 'latest_financial.csv'))
    fields = [c for c in financial if c not in ('name', 'industry', 'market', 'report_date', 'notice_date')]
    merged = snapshot.merge(financial[fields], on='code', how='left', validate='one_to_one')
    merged['scope'] = merged.industry.map(lambda x: 'core' if x in CORE else
                                         ('adjacent_review_required' if x in ADJACENT else 'outside'))
    merged['risk_name_flag'] = merged.name.str.contains('ST|退', case=False, na=False)
    merged['quote_valid'] = (merged.price > 0) & (merged.amount_yuan > 0)
    merged['user_board_eligible'] = ~merged.symbol.str.startswith('bj')
    merged.to_csv(str(BASE / 'all_a_screen_base.csv'), index=False, encoding='utf-8-sig')
    broad = merged[merged.scope != 'outside'].copy()
    broad.to_csv(str(BASE / 'broad_technology_financial.csv'), index=False, encoding='utf-8-sig')
    # No recent return, profit-growth, or valuation gate is used to select histories.
    history_pool = broad[(~broad.risk_name_flag) & broad.quote_valid].copy()
    history_pool.to_csv(str(BASE / 'technology_universe.csv'), index=False, encoding='utf-8-sig')
    meta = {'market_count': len(merged), 'financial_count': len(financial),
            'core_industries': CORE, 'adjacent_industries': ADJACENT,
            'core_count': int((broad.scope == 'core').sum()),
            'adjacent_count': int((broad.scope == 'adjacent_review_required').sum()),
            'broad_count': len(broad), 'historical_requests': len(history_pool),
            'name_risk_excluded': int(broad.risk_name_flag.sum()),
            'invalid_quote_count': int((~broad.quote_valid).sum()),
            'core_missing_financial': int(broad[broad.scope == 'core'].net_profit.isna().sum()),
            'bj_in_history_pool': int((~history_pool.user_board_eligible).sum()),
            'scope_note': 'Adjacent industry constituents are a coverage buffer, not a claim that every company is a technology business. Final candidates require business verification.'}
    (BASE / 'universe_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
