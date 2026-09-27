"""Shared, read-only three-version rendering, including old saved cards."""
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import push_signal
from push_signal import build_markdown, is_current_signal

NOW = datetime(2026, 9, 28, 14, 50, 30)
DATE = '2026-09-28'
HEADER = ['动量轮动信号 | 生成 2026-09-28 14:50:10 | 数据截止 2026-09-28',
          '行情时间: 2026-09-28 14:50:00']
LEGACY = HEADER + ['当前持仓: 518880 黄金ETF', '牛熊体制: 牛市（不应再显示）',
    '★ 建议: 卖出 518880 黄金ETF -> 买入 510300 沪深300ETF',
    '依据: 旧无版本主建议', '159915 创业板ETF MOM20 +10.00%',
    '513100 纳指ETF 溢价 +0.10%', '513120 港股创新药ETF 溢价 +8.00%']


def block(version, code='513100', name='纳指ETF', status=None, date=DATE, reason='示例原因', action=''):
    lines = ['-' * 56, '【影子 %s】虚拟跟踪不下单' % version]
    if status is not None: lines.append('信号状态: ' + status)
    if date is not None: lines.append('数据截止: ' + date)
    return lines + ['影子持仓: %s %s | 虚拟净值 1.0 | 建议: %s%s %s | %s' % (code,name,action,code,name,reason),
                    '跟踪说明: 原始份额记账']


def section(card, version):
    return card.split('**影子 %s**' % version, 1)[1].split('\n---', 1)[0]


class PushSignalTests(unittest.TestCase):
    def test_legacy_card_hides_primary_rank_and_0906_in_both_modes(self):
        signal = '\n'.join(LEGACY + block('v9.1-0906', '512400', '有色金属ETF', reason='旧0906原因')
                           + block('v9.2', '513120', '港股创新药ETF') + block('V10-H'))
        for query in (False, True):
            card, date = build_markdown(signal, query=query, now=NOW)
            self.assertEqual(date, DATE)
            for hidden in ('当前持仓:', '牛熊体制:', '旧无版本主建议', '动量前三', '创业板ETF MOM20', '0906', '旧0906原因', '买入 **510300'):
                self.assertNotIn(hidden, card)
            self.assertEqual(re.findall(r'\*\*影子 ([^*]+)\*\*', card), ['V9.2', 'V9.2+', 'V10-H'])
            self.assertIn('影子建议标的: **513120 港股创新药ETF**', section(card, 'V9.2'))
            self.assertIn('尚无有效信号；等待首次生成', section(card, 'V9.2+'))
            self.assertNotIn('513120', section(card, 'V9.2+'))
            self.assertNotIn('虚拟净值', section(card, 'V9.2+'))
            self.assertIn('0.10%', card); self.assertIn('8.00%', card)

    def test_fixed_order_independent_targets(self):
        signal = '\n'.join(LEGACY + block('V10-H', '518880', '黄金ETF')
                           + block('V9.2+', '510500', '中证500ETF') + block('v9.2'))
        card, unused = build_markdown(signal, now=NOW)
        self.assertEqual(re.findall(r'\*\*影子 ([^*]+)\*\*', card), ['V9.2', 'V9.2+', 'V10-H'])
        for version,target in (('V9.2','513100 纳指ETF'), ('V9.2+','510500 中证500ETF'), ('V10-H','518880 黄金ETF')):
            self.assertIn('影子建议标的: **' + target + '**', section(card, version))

    def test_new_card_uses_existing_store_freshness_contract(self):
        signal = '\n'.join(HEADER + block('V9.2') + block('V9.2+') + block('V10-H'))
        info = dict(date=DATE, actionable=True, note='当日有效信号', quote_time=DATE+' 14:50:00')
        with patch.object(push_signal.store, 'signal_info', return_value=info) as freshness:
            card, date = build_markdown(signal, now=NOW)
        freshness.assert_called_once_with(signal, now=NOW)
        self.assertEqual(date, DATE)
        self.assertEqual(card.count('影子建议标的:'), 3)
        self.assertNotIn('★ 建议', card)

    def test_v10_preview_stays_historical_even_when_outer_card_is_fresh(self):
        signal = '\n'.join(LEGACY + block('v9.2') + block('V10-H', status='历史预览；前向跟踪尚未启动',
            date='2026-09-24', reason='冻结研究历史收盘目标，非今日买入指令'))
        for query in (False, True):
            card, unused = build_markdown(signal, query=query, now=NOW)
            h = section(card, 'V10-H')
            for expected in ('历史影子目标: **513100 纳指ETF**', '历史预览；前向跟踪尚未启动',
                             '非今日买入指令', '数据截止: 2026-09-24'):
                self.assertIn(expected, h)
            self.assertNotIn('影子建议标的:', h)

    def test_block_date_mismatch_cannot_be_promoted_by_fresh_outer_header(self):
        signal = '\n'.join(LEGACY + block('V9.2', date='2026-09-24', action='买入 '))
        card, unused = build_markdown(signal, now=NOW)
        self.assertIn('历史影子目标:', section(card, 'V9.2'))
        self.assertNotIn('买入 **', section(card, 'V9.2'))

    def test_failed_section_suppresses_old_target_still_in_text(self):
        signal = '\n'.join(LEGACY + block('V10-H', status='计算失败，本次无有效建议', action='买入 ')
                           + ['数据说明: 缺当日报价'])
        card, unused = build_markdown(signal, now=NOW)
        h = section(card, 'V10-H')
        self.assertIn('本次无有效建议', h); self.assertIn('缺当日报价', h)
        for forbidden in ('影子建议标的:', '历史影子目标:', '买入 **'):
            self.assertNotIn(forbidden, h)

    def test_missing_empty_waiting_sections_never_invent_targets_or_nav(self):
        for sample in ('', '【影子 V9.2+】', '【影子 V9.2+】\n信号状态: 尚无有效信号；等待首次生成'):
            card, unused = build_markdown('\n'.join(LEGACY) + '\n' + sample, now=NOW)
            value = section(card, 'V9.2+')
            self.assertIn('等待首次生成', value)
            for forbidden in ('影子建议标的:', '虚拟净值', '计算失败'):
                self.assertNotIn(forbidden, value)

    def test_untitled_failure_belongs_only_to_exact_plus_version(self):
        signal = '\n'.join(LEGACY + block('v9.2') + ['-'*56, '⚠️ 影子 V9.2+ 计算失败: 缺少实时数据'])
        card, unused = build_markdown(signal, now=NOW)
        self.assertIn('影子建议标的:', section(card, 'V9.2'))
        self.assertIn('缺少实时数据', section(card, 'V9.2+'))
        self.assertNotIn('影子建议标的:', section(card, 'V9.2+'))

    def test_later_failure_cannot_fall_back_to_earlier_success(self):
        signal = '\n'.join(LEGACY + block('V9.2+') + ['【影子 V9.2+】', '信号状态: 计算失败，本次无有效建议'])
        card, unused = build_markdown(signal, now=NOW)
        self.assertEqual(card.count('**影子 V9.2+**'), 1)
        self.assertNotIn('影子建议标的:', section(card, 'V9.2+'))

    def test_qvix_channel_failure_does_not_disable_valid_strategy_target(self):
        signal = '\n'.join(LEGACY + block('V9.2') + ['QVIX 当日缺失，QVIX通道已停用；深跌/量能通道仍可用'])
        card, unused = build_markdown(signal, now=NOW)
        self.assertIn('影子建议标的:', section(card, 'V9.2'))
        self.assertIn('QVIX通道已停用', section(card, 'V9.2'))

    def test_buy_hold_and_held_assets_are_bold(self):
        signal = '\n'.join(LEGACY + block('V9.2', action='买入 ')
                           + block('V9.2+', '510500', '中证500ETF', action='继续持有 '))
        card, unused = build_markdown(signal, now=NOW)
        for expected in ('影子建议标的: 买入 **513100 纳指ETF**',
                         '影子建议标的: 继续持有 **510500 中证500ETF**', '影子持仓: **510500 中证500ETF**'):
            self.assertIn(expected, card)

    def test_global_stale_or_after_window_remains_historical(self):
        signal = '\n'.join(LEGACY + block('V9.2', action='买入 '))
        for now in (datetime(2026,9,29,14,50), datetime(2026,9,28,15,1)):
            card, unused = build_markdown(signal, query=True, now=now)
            self.assertIn('历史影子目标:', card)
            self.assertNotIn('影子建议标的:', card); self.assertNotIn('买入 **', card)
            self.assertIn('最近保存信号', card); self.assertIn('动量跟踪网页', card)

    def test_legacy_date_helper_unchanged(self):
        header = '动量轮动信号 | 生成 2026-09-24 14:50 | 数据截止 '
        self.assertTrue(is_current_signal(header+'2026-09-24', '2026-09-24'))
        self.assertFalse(is_current_signal(header+'2026-09-23', '2026-09-24'))
        self.assertFalse(is_current_signal('missing header', '2026-09-24'))

    def test_migration_csv_premium_delivery_notices_are_visible(self):
        note = '⚠️ 旧口径净值 1.1 已归档，2026-09-24 起重新从1记账'
        signal = '\n'.join(LEGACY + block('v9.2') + [note,
            '跟踪说明: CSV导出失败，JSON已入账', 'QDII 溢价: 本次获取失败'])
        card, unused = build_markdown(signal, now=NOW, notices=['自动推送尚未确认送达'])
        for expected in (note, 'CSV导出失败', 'QDII 溢价: 本次获取失败', '自动推送尚未确认送达'):
            self.assertIn(expected, card)

    def test_saved_query_filters_legacy_file_without_changing_bytes_or_sending(self):
        signal = '\n'.join(LEGACY + block('v9.1-0906', '512400', '有色金属ETF') + block('v9.2'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/'latest.txt'; path.write_text(signal)
            (root/'generation_status.json').write_text(json.dumps(dict(date=DATE, status='failed')))
            (root/'delivery.json').write_text(json.dumps(dict(signal_id=push_signal.store.digest(signal), status='uncertain')))
            before = {p.name:p.read_bytes() for p in root.iterdir()}
            with patch.object(push_signal.urllib.request, 'urlopen', side_effect=AssertionError('render cannot send')):
                card = push_signal.render_saved_query(str(path), now=NOW)
            self.assertNotIn('0906', card); self.assertNotIn('旧无版本主建议', card)
            self.assertIn('今日信号生成失败', card); self.assertIn('尚未确认送达', card)
            self.assertIn('等待首次生成', section(card, 'V9.2+'))
            self.assertEqual(before, {p.name:p.read_bytes() for p in root.iterdir()})

    def test_python36_import_and_render_when_available(self):
        binary = shutil.which('python3.6')
        if not binary: self.skipTest('Python3.6 executable unavailable')
        code = "import push_signal; text,date=push_signal.build_markdown(''); assert 'V9.2+' in text"
        done = subprocess.run([binary, '-B', '-c', code], cwd=push_signal.BASE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=10)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == '__main__': unittest.main()
