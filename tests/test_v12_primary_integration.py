"""Four-account transaction and bot compatibility; all state/network is isolated."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import push_signal
import signal_daily
import signal_store as store
from market_data import UNIVERSE
from v10_live import ledger as h_ledger
from v92_plus_live import ledger as plus_ledger
from v12_live import ledger as r2_ledger

DAY = "2026-09-29"
NOW = datetime(2026, 9, 29, 14, 50, 30)
PRIMARY_CODE = "518880"
COMPARISON_CODE = "513100"
PRIMARY = "shadow_v12_r2.json"
NAMES = (PRIMARY, "shadow_v92.json", "shadow_v92_plus.json", "shadow_v10.json")
LABELS = dict(zip(NAMES, ("V12-R2", "V9.2", "V9.2+", "V10-H")))


def quotes(day):
    return {code: dict(date=day,timestamp=day+" 14:50:00",price=10.,open=10.,volume=100.,prev_close=10.)
            for code in UNIVERSE}


def card(name, state):
    code = state["holding"]
    return ["-"*56, "【影子 %s】独立虚拟跟踪" % LABELS[name], "信号状态: 盘中影子试算",
            "数据截止: %s | 行情时间: %s" % (state["last_date"], state["quote_timestamp"]),
            "影子持仓: %s | 虚拟净值 %.4f | 建议: %s %s | synthetic" %
            (code,state["nav"],code,UNIVERSE[code][0])]


def account(name, day=DAY):
    code = PRIMARY_CODE if name == PRIMARY else COMPARISON_CODE
    if name == "shadow_v92.json":
        value = dict(valuation_version=4,account_id="v9.2",holding=code,nav=1.,units=.1,cash=0.,
            mark_raw_price=10.,last_date=day,start_date=day,entry_date=day,lock_code=None,
            quote_timestamp=day+" 14:50:00",
            events=[dict(date=day,to=code,nav=1.,baseline=True,trade=False,price=10.,**{"from":None})])
    else:
        ledger = {PRIMARY:r2_ledger,"shadow_v92_plus.json":plus_ledger,"shadow_v10.json":h_ledger}[name]
        decision = dict(candidate_id=ledger.CANDIDATE_ID,candidate_hash=ledger.CANDIDATE_HASH,
                        target=code,executable=True,reason="synthetic")
        if name == PRIMARY: decision["strategy_id"] = r2_ledger.STRATEGY_ID
        view = dict(calendar=[day],actions={c:{day:dict(split_ratio=1.,cash_per_old_share=0.)} for c in UNIVERSE})
        value = ledger.advance(None,decision,view,quotes(day),day)
    value["saved_lines"] = card(name,value)
    return value


def main_card(text):
    return text.split("**主推送 V12-R2**",1)[1].split("\n---",1)[0]


class PrimaryTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.moment = [NOW]
        self.successful = list(NAMES)
        self.stale_primary = False
        self.stack.enter_context(patch.object(signal_daily,"SIGNAL_DIR",str(self.root)))
        self.stack.enter_context(patch.object(store,"now_local",side_effect=lambda:self.moment[0]))
        self.fetch = self.stack.enter_context(patch.object(signal_daily,"fetch_realtime",side_effect=lambda **kwargs:quotes(self.moment[0].strftime("%Y-%m-%d"))))
        self.prepare = self.stack.enter_context(patch.object(signal_daily,"prepare_versions",side_effect=self.worker))
        self.stack.enter_context(patch.object(push_signal.urllib.request,"urlopen",side_effect=AssertionError("No network")))
        self.stack.enter_context(patch("sys.stdout",new_callable=io.StringIO))

    def worker(self, unused_quotes, day, stage, now=None):
        lines = []
        # The parent owns all four real account locks before the worker begins.
        for name in NAMES:
            with self.assertRaises(BlockingIOError):
                with store.file_lock((self.root/name).with_suffix(".lock")):
                    self.fail("Unprotected real account")
        for name in NAMES:
            if name in self.successful:
                state = account(name,day)
                if name == PRIMARY and self.stale_primary:
                    state["quote_timestamp"] = day+" 14:45:00"
                    state["saved_lines"] = card(name,state)
                (Path(stage)/name).write_text(json.dumps(state))
                lines += state["saved_lines"]
            else:
                lines += ["-"*56,"【影子 %s】" % LABELS[name],"信号状态: 计算失败，本次无有效建议"]
        return dict(lines=lines,successful_accounts=list(self.successful),diagnostics=[])

    def run_main(self):
        return signal_daily.main(now=self.moment[0])

    def test_four_account_commit_uses_only_new_primary_and_cold_nav_is_one(self):
        self.assertEqual(self.run_main(),PRIMARY_CODE)
        saved = store.load_journal(self.root)
        self.assertEqual(saved["primary_version"],store.PRIMARY_VERSION)
        self.assertEqual(saved["target"],PRIMARY_CODE)
        self.assertEqual(store.primary_target(saved),PRIMARY_CODE)
        self.assertEqual(set(saved["updated_accounts"]),set(NAMES))
        new = saved["account_states"][PRIMARY]
        self.assertEqual(new["nav"],1.)
        self.assertEqual(new["start_date"],DAY)
        self.assertEqual(new["candidate_id"],r2_ledger.CANDIDATE_ID)
        self.assertNotEqual(new["candidate_id"],saved["account_states"]["shadow_v10.json"]["candidate_id"])
        card_text,_ = push_signal.build_markdown(saved["text"],now=NOW)
        main = main_card(card_text)
        self.assertIn("主推送标的: **518880",main)
        self.assertNotIn("513100",main)
        self.assertLess(card_text.index("**主推送 V12-R2**"),card_text.index("**影子 V9.2**"))
        self.fetch.side_effect = AssertionError("Same-day fetch is not allowed")
        self.prepare.side_effect = AssertionError("Same-day rebooking is not allowed")
        before = {name:(self.root/name).read_bytes() for name in NAMES}
        self.assertEqual(self.run_main(),PRIMARY_CODE)
        self.assertEqual(before,{name:(self.root/name).read_bytes() for name in NAMES})

    def test_old_three_account_journal_replays_without_new_trade_or_legacy_primary(self):
        states = {name:account(name) for name in NAMES if name != PRIMARY}
        text = "\n".join(["动量轮动信号 | 生成 %s 14:50:10 | 数据截止 %s" % (DAY,DAY),
                         "行情时间: %s 14:50:00" % DAY,
                         "策略版本: V9.2 | V9.2+ | V10-H"] +
                        [line for state in states.values() for line in state["saved_lines"]])
        old,_ = store.publish(text,DAY,COMPARISON_CODE,None,self.root,account_states=states)
        self.assertNotIn("primary_version",old)
        before = (self.root/"daily_state.json").read_bytes()
        self.fetch.side_effect = AssertionError("Must replay old committed bundle")
        self.prepare.side_effect = AssertionError("Must not backfill a newly deployed account")
        self.assertIsNone(self.run_main())
        self.assertEqual((self.root/"daily_state.json").read_bytes(),before)
        self.assertFalse((self.root/PRIMARY).exists())
        rendered,_ = push_signal.build_markdown(text,now=NOW)
        self.assertIn("主推送本次无有效建议",rendered)
        self.assertNotIn("513100",main_card(rendered))
        self.assertIn("影子建议标的:",rendered)
        self.assertNotIn("✅",rendered)

    def test_failed_new_primary_keeps_checkpoint_but_cannot_fall_back_to_comparisons(self):
        self.run_main()
        old = deepcopy(store.load_journal(self.root)["account_states"][PRIMARY])
        primary_bytes = (self.root/PRIMARY).read_bytes()
        self.moment[0] = NOW.replace(day=30)
        self.successful = list(NAMES[1:])
        self.assertIsNone(self.run_main())
        record = store.load_journal(self.root)
        self.assertIsNone(record["target"])
        self.assertIsNone(store.primary_target(record))
        self.assertEqual(record["account_states"][PRIMARY],old)
        self.assertNotIn(PRIMARY,record["updated_accounts"])
        self.assertEqual((self.root/PRIMARY).read_bytes(),primary_bytes)
        rendered,_ = push_signal.build_markdown(record["text"],now=self.moment[0])
        self.assertIn("无有效建议",main_card(rendered))
        self.assertNotIn("主推送标的:",rendered)
        self.assertIn("影子建议标的:",rendered)
        self.assertNotIn("✅",rendered)

    def test_failed_canonical_commit_changes_none_of_the_four_real_accounts(self):
        before = {}
        for name in NAMES:
            (self.root/name).write_text(json.dumps(dict(original_marker=name)))
            before[name] = (self.root/name).read_bytes()
        original = store.atomic_json
        def fail(path,value,**kwargs):
            if Path(path).name == "daily_state.json":raise OSError("Synthetic canonical failure")
            return original(path,value,**kwargs)
        with patch.object(store,"atomic_json",side_effect=fail):
            with self.assertRaises(OSError):self.run_main()
        self.assertFalse((self.root/"daily_state.json").exists())
        self.assertEqual(before,{name:(self.root/name).read_bytes() for name in NAMES})
        self.assertFalse(list(self.root.glob(".signal-stage-*")))

    def test_missing_primary_projection_recovers_from_committed_checkpoint_without_retrading(self):
        original = store.atomic_text
        def fail(path,text,**kwargs):
            if Path(path) == self.root/PRIMARY:raise OSError("Synthetic primary projection failure")
            return original(path,text,**kwargs)
        with patch.object(store,"atomic_text",side_effect=fail):
            self.assertEqual(self.run_main(),PRIMARY_CODE)
        self.assertFalse((self.root/PRIMARY).exists())
        committed = store.load_journal(self.root)
        self.assertEqual(store.primary_target(committed),PRIMARY_CODE)
        self.fetch.side_effect = AssertionError("Projection repair must not refetch")
        self.prepare.side_effect = AssertionError("Projection repair must not trade")
        self.assertEqual(self.run_main(),PRIMARY_CODE)
        self.assertEqual(json.loads((self.root/PRIMARY).read_text()),committed["account_states"][PRIMARY])

    def test_individually_stale_primary_quote_blocks_the_whole_proposed_commit(self):
        self.stale_primary = True
        with self.assertRaisesRegex(ValueError,"行情已过期"):
            self.run_main()
        self.assertFalse((self.root/"daily_state.json").exists())
        self.assertTrue(all(not (self.root/name).exists() for name in NAMES))

    def test_primary_identity_cannot_claim_comparison_target(self):
        states = {name:account(name) for name in NAMES}
        with self.assertRaises(ValueError):
            store.publish("synthetic",DAY,COMPARISON_CODE,None,self.root,account_states=states,
                          primary_version=store.PRIMARY_VERSION)
        self.assertFalse((self.root/"daily_state.json").exists())
        self.run_main()
        value = store.load_journal(self.root)
        value["target"] = COMPARISON_CODE
        value["checksum"] = store.record_digest(value)
        (self.root/"daily_state.json").write_text(json.dumps(value))
        with self.assertRaises(ValueError):store.load_journal(self.root)

    def test_rotation_and_crash_rule_text_keep_action_meaning_for_current_history_and_failure(self):
        lines = ["动量轮动信号 | 生成 %s 14:50:10 | 数据截止 %s" % (DAY,DAY),
                 "行情时间: %s 14:50:00" % DAY,
                 "策略版本: V12-R2 | V9.2 | V9.2+ | V10-H"] + account(PRIMARY)["saved_lines"]
        lines += ["抄底规则: 深跌与量能通道；原5日锁仓优先于急跌退出",
                  "换仓动作: 卖出 513100 纳指ETF，买入 518880 黄金ETF"]
        text = "\n".join(lines)
        current,_ = push_signal.build_markdown(text,now=NOW)
        main = main_card(current)
        self.assertIn("抄底规则: 深跌与量能通道",main)
        self.assertIn("卖出 513100 纳指ETF，买入 **518880 黄金ETF**",main)
        self.assertNotIn("卖出 **513100",main)
        history,_ = push_signal.build_markdown(text,now=NOW.replace(hour=15))
        historical_main = main_card(history)
        self.assertIn("历史虚拟换仓记录:",historical_main)
        self.assertIn("非当前操作指令",historical_main)
        self.assertNotIn("买入 **",historical_main)
        failed,_ = push_signal.build_markdown(text+"\n信号状态: 计算失败，本次无有效建议",now=NOW)
        self.assertNotIn("换仓动作:",main_card(failed))
        self.assertNotIn("主推送标的:",main_card(failed))

    def test_python36_bot_can_read_four_account_journal_from_foreign_directory(self):
        binary = shutil.which("python3.6")
        if not binary:self.skipTest("Python3.6 unavailable")
        self.run_main()
        before = {p.name:p.read_bytes() for p in self.root.iterdir() if p.is_file()}
        script = "\n".join([
            "import importlib.util",
            "from datetime import datetime",
            "spec=importlib.util.spec_from_file_location('formatter',%r)" % str(Path(push_signal.__file__).resolve()),
            "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)",
            "text=module.render_saved_query(path=%r,now=datetime(2026,9,29,14,50,30))" % str(self.root/"latest.txt"),
            "assert '主推送标的: **518880' in text",
            "assert '**影子 V9.2**' in text and '**影子 V10-H**' in text",
        ])
        done = subprocess.run([binary,"-B","-c",script],cwd=str(self.root),stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,universal_newlines=True,timeout=10)
        self.assertEqual(done.returncode,0,done.stderr)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir() if p.is_file()})

    def test_actual_four_adapters_share_json_inputs_and_write_only_staging(self):
        import daily_extras
        import premium
        import shadow_0906
        import shadow_v92
        from v10_live import data, runtime as h_runtime
        from v92_plus_live import runtime as plus_runtime
        from v12_live import runtime as r2_runtime
        seed = data._load_seed()
        pairs, current_quotes = {}, {}
        for code in data.CODES:
            with (data.SEED_DIR/"snapshots"/(code+".csv")).open(newline="") as stream:
                adjusted = {r[0]:(r[0],float(r[1]),float(r[2]),float(r[3])) for r in csv.reader(stream) if r}
            history = seed["raw"][code][-22:]
            price, volume = float(history[-1][2]), float(history[-1][3])*.8
            row = (DAY,price,price,volume)
            pairs[code] = dict(raw=history+[row],qfq=[adjusted[r[0]] for r in history]+[row],
                               sources={"raw":"synthetic","qfq":"same-vintage offline"})
            current_quotes[code] = dict(date=DAY,timestamp=DAY+" 14:50:00",price=price,open=price,
                                       volume=volume,prev_close=price)
        view,unused = data._build(seed,data._empty_state(seed),pairs,data._quotes(current_quotes,DAY,NOW),DAY,None)
        stage = self.root/"stage";stage.mkdir()
        qvix = self.root/"qvix50.csv"
        qvix.write_bytes((data.SEED_DIR/"snapshots/qvix50.csv").read_bytes())
        production = self.root/"untouched-production";production.mkdir()
        before = {}
        for name in NAMES:
            (production/name).write_text(json.dumps({"marker":name}))
            before[name] = (production/name).read_bytes()
        with ExitStack() as stack:
            stack.enter_context(patch.object(data,"build_live_view",return_value=view))
            stack.enter_context(patch.object(shadow_0906,"qvix_state",return_value=(None,None,None,False,"offline")))
            stack.enter_context(patch.object(shadow_0906,"QVIX_CSV",str(qvix)))
            stack.enter_context(patch.object(premium,"signal_block",return_value=[]))
            stack.enter_context(patch.object(shadow_v92,"STATE_FILE",str(production/"shadow_v92.json")))
            stack.enter_context(patch.object(shadow_v92,"TRADES_FILE",str(production/"shadow_v92_trades.csv")))
            stack.enter_context(patch.object(shadow_v92,"PORTFOLIO",str(production/"unused_portfolio.json")))
            for module,name in ((h_runtime,"shadow_v10.json"),(plus_runtime,"shadow_v92_plus.json"),(r2_runtime,PRIMARY)):
                original = module.run
                def guarded(*args,_original=original,_name=name,**kwargs):
                    self.assertEqual(kwargs.get("state_path"),stage/_name)
                    return _original(*args,**kwargs)
                stack.enter_context(patch.object(module,"run",side_effect=guarded))
            result = daily_extras.collect(dict(quotes=current_quotes,date=DAY,state_dir=str(stage),now=NOW.isoformat()))
        self.assertEqual(result["successful_accounts"],list(NAMES),result.get("diagnostics"))
        for name in NAMES:
            value = json.loads((stage/name).read_text())
            self.assertEqual(value["last_date"],DAY)
            self.assertEqual(value["nav"],1.)
            self.assertEqual(value["start_date"],DAY)
        main = json.loads((stage/PRIMARY).read_text())
        self.assertEqual(main["strategy_id"],r2_ledger.STRATEGY_ID)
        self.assertEqual(main["candidate_id"],r2_ledger.CANDIDATE_ID)
        self.assertEqual(before,{name:(production/name).read_bytes() for name in NAMES})
        self.assertFalse((production/"shadow_v92_trades.csv").exists())


if __name__=="__main__":unittest.main()
