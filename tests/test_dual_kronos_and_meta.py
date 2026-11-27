import ast
import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.stubs import FakeDataFrame, install_missing_stubs

install_missing_stubs()

import bot_state
import config
import finetune_data
import hf_repo_sync
import history_tracker
import kronos_loader
import kronos_predictor
import main
import meta_model

VN_TZ = timezone(timedelta(hours=7))


def _make_clean_signal(idx, hit_tp=True, high_quality=True):
    conf = 88.0 if high_quality else 75.5
    agree = 100.0 if high_quality else 60.0
    adx = 38.0 if high_quality else 21.0
    std_pct = 0.08 if high_quality else 0.65
    tp_dist = 900.0 if high_quality else 360.0
    return {
        "id": f"BTCUSD_20261101_{idx:06d}",
        "symbol": "BTCUSD" if idx % 2 == 0 else "XAUUSD",
        "timeframe": "15m" if idx % 2 == 0 else "1h",
        "direction": "🟢 BUY (TĂNG)",
        "entry_price": 70000.0,
        "tp": 70000.0 + tp_dist,
        "sl": 69700.0,
        "open_time": f"2026-11-{(idx % 25) + 1:02d}T10:00:00+00:00",
        "open_time_vn": f"2026-11-{(idx % 25) + 1:02d} 17:00:00",
        "status": "HIT_TP" if hit_tp else "HIT_SL",
        "closed_at": f"2026-11-{(idx % 25) + 1:02d}T12:00:00+00:00",
        "close_price": (70000.0 + tp_dist) if hit_tp else 69700.0,
        "confidence": conf,
        "pct_change": 1.35 if high_quality else 0.56,
        "ref_close": 69990.0,
        "atr": 200.0,
        "adx": adx,
        "ema_50": 69600.0 if high_quality else 69985.0,
        "ema_200": 69000.0,
        "sample_agreement": agree,
        "sample_std_pct": std_pct,
        "strategy_version": "v3.3-p1-audit" if idx % 3 == 0 else "v4.0-dual-kronos-meta",
    }


def _make_bars(count=45, base_price=70000.0, step_price=40.0):
    t0 = datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc)
    times = [(t0 + timedelta(minutes=15 * i)).isoformat() for i in range(count)]
    closes = [base_price + step_price * i for i in range(count)]
    return FakeDataFrame({
        "time": times,
        "open": [c - 10.0 for c in closes],
        "high": [c + 80.0 for c in closes],
        "low": [c - 80.0 for c in closes],
        "close": closes,
        "volume": [120.0 + i for i in range(count)],
    })


class TestDualKronosAndMeta(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.orig_hist = history_tracker.HISTORY_FILE
        self.orig_state = bot_state.STATE_FILE
        history_tracker.HISTORY_FILE = os.path.join(self.tmp_dir.name, "signals_history.json")
        bot_state.STATE_FILE = os.path.join(self.tmp_dir.name, "bot_state.json")
        kronos_predictor._GLOBAL_PREDICTOR = None
        kronos_loader.reset_loader_cache()

    def tearDown(self):
        history_tracker.HISTORY_FILE = self.orig_hist
        bot_state.STATE_FILE = self.orig_state
        kronos_predictor._GLOBAL_PREDICTOR = None
        kronos_loader.reset_loader_cache()
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        self.tmp_dir.cleanup()

    def test_dual_branch_loading_shares_tokenizer_and_separates_branches(self):
        tok_calls, mod_calls = [], []

        class DummyObj:
            def eval(self):
                return self

        class DummyPredictor:
            def __init__(self, model, tokenizer, **kwargs):
                self.model = model
                self.tokenizer = tokenizer

        fake_pkg = types.ModuleType("model")
        fake_pkg.KronosTokenizer = types.SimpleNamespace(
            from_pretrained=lambda repo, **kw: (tok_calls.append((repo, kw)) or DummyObj())
        )
        fake_pkg.Kronos = types.SimpleNamespace(
            from_pretrained=lambda repo, **kw: (mod_calls.append((repo, kw)) or DummyObj())
        )
        fake_pkg.KronosPredictor = DummyPredictor

        with (
            mock.patch.dict(sys.modules, {"model": fake_pkg}),
            mock.patch.object(config, "HF_TOKEN", "hf_test_token"),
        ):
            btc_pred = kronos_predictor.load_kronos_base("BTCUSD", "btc-m15")
            xau_pred = kronos_predictor.load_kronos_base("XAUUSD", "xau-h1")
            btc_cached = kronos_predictor.load_kronos_base("BTCUSD", "btc-m15")

        self.assertIs(btc_pred, btc_cached)
        self.assertIsNot(btc_pred, xau_pred)
        self.assertEqual(len(tok_calls), 1)
        self.assertEqual(tok_calls[0][0], "kingescaped/lvq-std-BOT")
        self.assertEqual(tok_calls[0][1].get("revision"), "tokenizer")
        self.assertEqual(len(mod_calls), 2)
        self.assertEqual(btc_pred.model_source, "kingescaped/lvq-std-BOT@btc-m15")
        self.assertEqual(xau_pred.model_source, "kingescaped/lvq-std-BOT@xau-h1")

        kronos_loader.reset_loader_cache()
        tok_calls.clear()
        with mock.patch.object(config, "KRONOS_TOKENIZER_NAME", "custom/standalone-tok"):
            kronos_loader.load_tokenizer_instance(fake_pkg.KronosTokenizer, "hf_test_token")
        self.assertEqual(tok_calls[0][0], "custom/standalone-tok")
        self.assertIsNone(tok_calls[0][1].get("revision"))

    def test_fallback_uses_token_false_when_token_missing_or_expired(self):
        attempts = []

        class DummyObj:
            def eval(self):
                return self

        def load_tok(repo, **kw):
            eff_tok = os.environ.get("HF_TOKEN") if kw.get("token") is None else kw.get("token")
            if eff_tok == "expired_tok":
                raise RuntimeError("401 Invalid token")
            return DummyObj()

        def load_mod(repo, **kw):
            attempts.append((repo, kw.get("revision"), kw.get("token")))
            eff_tok = os.environ.get("HF_TOKEN") if kw.get("token") is None else kw.get("token")
            if repo == "kingescaped/lvq-std-BOT" or eff_tok == "expired_tok":
                raise RuntimeError("401/404 failure")
            return DummyObj()

        fake_pkg = types.ModuleType("model")
        fake_pkg.KronosTokenizer = types.SimpleNamespace(from_pretrained=load_tok)
        fake_pkg.Kronos = types.SimpleNamespace(from_pretrained=load_mod)
        fake_pkg.KronosPredictor = lambda model, tokenizer, **kw: types.SimpleNamespace(model=model)

        with (
            mock.patch.dict(sys.modules, {"model": fake_pkg}),
            mock.patch.object(config, "HF_TOKEN", "expired_tok"),
        ):
            pred = kronos_predictor.load_kronos_base("BTCUSD", "btc-m15")

        self.assertIsNotNone(pred)
        self.assertEqual(pred.model_source, "NeoQuasar/Kronos-base")
        self.assertEqual(attempts[-1], ("NeoQuasar/Kronos-base", None, False))

    def test_hf_repo_sync_creates_private_repo_and_syncs_missing_files(self):
        created_repos, created_branches, folder_uploads = [], [], []

        class FakeApi:
            def __init__(self, token=None):
                self.token = token

            def create_repo(self, **kw):
                created_repos.append(kw)

            def list_repo_files(self, repo_id, repo_type, revision):
                if revision == "main":
                    return ["config.json", "README.md"]
                return ["config.json", "README.md", "model.safetensors"] if revision == "btc-m15" else ["config.json"]

            def create_branch(self, **kw):
                created_branches.append(kw)

            def upload_file(self, **kw):
                pass

            def upload_folder(self, **kw):
                folder_uploads.append(kw["revision"])

        src_file = os.path.join(self.tmp_dir.name, "dummy_src.json")
        with open(src_file, "w", encoding="utf-8") as f:
            json.dump({"n_layers": 12}, f)

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.HfApi = FakeApi
        fake_hf.hf_hub_download = lambda repo_id, filename, repo_type, **kw: src_file

        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            summary = hf_repo_sync.sync_base_to_private_repo(token="hf_write_tok")

        self.assertTrue(created_repos[0]["private"])
        self.assertEqual([b["branch"] for b in created_branches], ["btc-m15", "xau-h1", "tokenizer"])
        self.assertEqual(summary["branches"]["main"], ["model.safetensors"])
        self.assertIn("model.safetensors", summary["branches"]["xau-h1"])
        self.assertEqual(folder_uploads, ["tokenizer"])

    def test_meta_shadow_mode_and_midweek_crossing_100_activates_immediately(self):
        legacy = [{"id": f"OLD_{i}", "symbol": "BTCUSD", "status": "HIT_SL", "entry_price": 60000.0, "tp": 61000.0, "sl": 59500.0} for i in range(120)]
        clean_30 = [_make_clean_signal(i, hit_tp=(i % 3 == 0), high_quality=(i % 3 == 0)) for i in range(30)]
        history_tracker.save_history(legacy + clean_30)

        state_30 = meta_model.train_weekly_meta_model(history_tracker.load_history())
        self.assertEqual(state_30["clean_trade_count"], 30)
        self.assertTrue(state_30["trained"])
        self.assertFalse(state_30["filter_active"])

        weak = {
            "current_price": 70000.0, "ref_close": 69990.0, "direction": "🟢 BUY (TĂNG)",
            "confidence": 75.2, "tp": 70360.0, "sl": 69700.0, "atr": 200.0, "adx": 20.5,
            "ema_50": 69988.0, "pct_change": 0.56, "sample_agreement": 60.0, "sample_std_pct": 0.70, "ai_predicted": True,
        }
        self.assertFalse(meta_model.evaluate_meta_signal("BTCUSD", weak, history=history_tracker.load_history()).blocked)

        clean_105 = [_make_clean_signal(i, hit_tp=(i < 50), high_quality=(i < 50)) for i in range(105)]
        history_tracker.save_history(legacy + clean_105)
        dec_midweek = meta_model.evaluate_meta_signal("BTCUSD", weak, history=history_tracker.load_history())
        self.assertTrue(dec_midweek.filter_active)
        self.assertTrue(dec_midweek.blocked)

    def test_meta_gate_in_main_and_feature_zero_nan_safety(self):
        import numpy as np

        feat_zero = meta_model.extract_meta_features(
            "BTCUSD",
            {"entry_price": 70000.0, "tp": 70800.0, "sl": 69600.0, "atr": 200.0, "ema_50": float("nan"), "sample_agreement": 0.0, "sample_std_pct": 0.0},
        )
        self.assertFalse(bool(np.isnan(feat_zero).any()))
        self.assertEqual(float(feat_zero[4]), 0.0)
        self.assertEqual(float(feat_zero[5]), 0.0)

        history_tracker.save_history([_make_clean_signal(i, hit_tp=(i < 55), high_quality=(i < 55)) for i in range(110)])
        meta_model.train_weekly_meta_model(history_tracker.load_history())
        weak = {
            "current_price": 70000.0, "ref_close": 69990.0, "direction": "🟢 BUY (TĂNG)",
            "confidence": 75.2, "tp": 70360.0, "sl": 69700.0, "atr": 200.0, "adx": 20.5,
            "ema_50": 69988.0, "pct_change": 0.56, "sample_agreement": 60.0, "sample_std_pct": 0.70, "ai_predicted": True,
        }
        df = _make_bars()
        with (
            mock.patch.object(main, "create_signal_chart", return_value=None),
            mock.patch.object(main, "send_telegram_message", return_value=True) as msg_spy,
        ):
            main._handle_symbol_result(config.SYMBOLS[0], df, "2026-11-27T10:00:00+00:00", weak, manual_force=False)
            self.assertEqual(msg_spy.call_count, 0)
            main._handle_symbol_result(config.SYMBOLS[0], df, "2026-11-27T10:00:00+00:00", weak, manual_force=True)
            self.assertEqual(msg_spy.call_count, 1)
            self.assertIn("Meta-Model chặn kèo", msg_spy.call_args.args[0])

    def test_saturday_report_trains_meta_and_module_causal_eval_guard(self):
        history_tracker.save_history([_make_clean_signal(i, hit_tp=(i % 2 == 0), high_quality=(i % 2 == 0)) for i in range(30)])
        with (
            mock.patch.object(main, "generate_daily_report", return_value="daily"),
            mock.patch.object(main, "generate_weekly_report", return_value="weekly"),
            mock.patch.object(main, "send_telegram_message", return_value=True) as send_spy,
        ):
            main.check_and_send_daily_report_if_due(now_vn=datetime(2026, 11, 28, 6, 3, tzinfo=VN_TZ))
        self.assertEqual(send_spy.call_count, 2)
        self.assertEqual(meta_model.load_meta_state().get("clean_trade_count"), 30)

        root = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(root, "model", "module.py"), "r", encoding="utf-8") as f:
            mod_src = f.read()
        self.assertIn("q_len == seq_len and q_len > 1", mod_src)

        with open(os.path.join(root, "colab", "finetune_kronos_monthly.ipynb"), "r", encoding="utf-8") as f:
            nb = json.load(f)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                cleaned_lines = [ln for ln in cell["source"] if not ln.lstrip().startswith(("!", "%"))]
                ast.parse("".join(cleaned_lines))

    def test_finetune_window_builder_has_no_train_val_overlap_or_future_norm_leakage(self):
        import numpy as np

        n = 700
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        times = [(t0 + timedelta(minutes=15 * i)).isoformat() for i in range(n)]
        prices = [50000.0 + i * 10.0 for i in range(n)]
        prices[-5:] = [9999999.0] * 5

        df = FakeDataFrame({
            "time": times, "open": prices, "high": prices, "low": prices,
            "close": prices, "volume": [10.0] * n, "amount": [500000.0] * n,
        })
        df.values = np.column_stack([prices, prices, prices, prices, [10.0] * n, [500000.0] * n])

        with (
            mock.patch.object(finetune_data, "calc_time_stamps", side_effect=lambda ts: types.SimpleNamespace(values=np.zeros((len(ts), 5), dtype=np.float32))),
            mock.patch.object(finetune_data.pd, "to_datetime", side_effect=lambda col, utc=True: col, create=True),
        ):
            orig_getitem = FakeDataFrame.__getitem__
            try:
                FakeDataFrame.__getitem__ = lambda s, k: (types.SimpleNamespace(values=df.values) if isinstance(k, list) else orig_getitem(s, k))
                (train_x, _), (val_x, _) = finetune_data.build_kronos_datasets(df, context_len=200, pred_len=12, stride=10)
            finally:
                FakeDataFrame.__getitem__ = orig_getitem

        self.assertGreater(len(train_x), 10)
        self.assertGreater(len(val_x), 2)
        self.assertEqual(train_x.shape[1], 213)
        self.assertAlmostEqual(float(np.mean(val_x[-1, :200, 0])), 0.0, places=3)

        import finetune_trainer
        ft_default = finetune_trainer.FineTuneConfig(symbol="BTCUSD", branch="btc-m15")
        self.assertEqual(ft_default.learning_rate, 1e-6)
        self.assertEqual(ft_default.weight_decay, 0.1)
        self.assertEqual(ft_default.adam_betas, (0.9, 0.95))
        self.assertEqual(ft_default.max_grad_norm, 3.0)


if __name__ == "__main__":
    unittest.main()
