from datetime import datetime, timedelta, timezone
import os
import sys
import types
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.stubs import FakeDataFrame, install_missing_stubs

install_missing_stubs()

import finetune_data
import finetune_trainer


def _build_synthetic_candles(days, step_minutes):
    bars_per_day = int((24 * 60) / step_minutes)
    total = days * bars_per_day
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    times = [(t0 + timedelta(minutes=step_minutes * i)).isoformat() for i in range(total)]
    closes = [2500.0 + 0.05 * i for i in range(total)]
    return FakeDataFrame({
        "time": times,
        "open": closes,
        "high": [c + 2.0 for c in closes],
        "low": [c - 2.0 for c in closes],
        "close": closes,
        "volume": [100.0] * total,
        "amount": [250000.0] * total,
    })


class _DummyParam:
    def __init__(self):
        self.device = "cpu"
        self.requires_grad = True


class _DummyModel:
    def __init__(self, tag):
        self.tag = tag
        self.saved_dirs = []
        self._param = _DummyParam()
        self._state = {"tag": tag, "epoch": 0}

    def to(self, device):
        self._param.device = device
        return self

    def parameters(self):
        return iter([self._param])

    def eval(self):
        return self

    def train(self):
        return self

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state):
        self._state = dict(state)

    def save_pretrained(self, folder):
        self.saved_dirs.append(folder)


class TestMonthlyFineTuneUpgrade(unittest.TestCase):
    def test_splits_exact_30_days_for_btc_365d_and_gold_730d(self):
        gold_df = _build_synthetic_candles(days=730, step_minutes=60)
        btc_df = _build_synthetic_candles(days=365, step_minutes=60)

        with mock.patch.object(
            finetune_data,
            "calc_time_stamps",
            side_effect=lambda ts: types.SimpleNamespace(values=np.zeros((len(ts), 5), dtype=np.float32)),
        ):
            (gold_train_x, _), (gold_val_x, _) = finetune_data.build_kronos_datasets(
                gold_df, context_len=50, pred_len=6, stride=10, val_days=30
            )
            (btc_train_x, _), (btc_val_x, _) = finetune_data.build_kronos_datasets(
                btc_df, context_len=50, pred_len=6, stride=10, val_days=30
            )

        expected_30d_windows = len(range(24 * 700 - 50, 24 * 730 - 57 + 1, 10))
        self.assertAlmostEqual(len(gold_val_x), expected_30d_windows, delta=2)
        self.assertAlmostEqual(len(btc_val_x), expected_30d_windows, delta=2)
        self.assertGreater(len(gold_train_x), len(btc_train_x))

    def test_validates_recent_candle_density_and_corrupted_ohlc(self):
        df_ok = _build_synthetic_candles(days=60, step_minutes=60)
        recent_bars = finetune_data.validate_recent_candles(df_ok, val_days=30, min_bars=700)
        self.assertGreaterEqual(recent_bars, 720)

        with self.assertRaises(ValueError):
            finetune_data.validate_recent_candles(df_ok, val_days=30, min_bars=900)

        bad_df = _build_synthetic_candles(days=40, step_minutes=60)
        bad_high = list(bad_df._data["low"])
        bad_high[-1] -= 10.0
        bad_df["high"] = bad_high
        with self.assertRaises(ValueError):
            finetune_data.validate_recent_candles(bad_df, val_days=30, min_bars=100)

    def test_recency_weights_monotonically_favor_newest_windows(self):
        weights = finetune_data.compute_recency_weights(100, min_weight=0.5, max_weight=1.5)
        self.assertEqual(len(weights), 100)
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=6)
        self.assertAlmostEqual(float(weights[-1] / weights[0]), 3.0, places=5)
        self.assertTrue(bool((np.diff(weights) > 0).all()))

    def test_judge_model_update_handles_win_tolerance_loss_and_anomalies(self):
        cfg = finetune_trainer.FineTuneConfig(symbol="BTCUSD", branch="btc-m15", val_tolerance=0.015, anomaly_val_loss=3.0)

        win = finetune_trainer.judge_model_update(2.50, 2.44, cfg)
        self.assertEqual((win.status, win.accepted, win.anomaly), ("UPDATED", True, False))

        within_tol = finetune_trainer.judge_model_update(2.50, 2.512, cfg)
        self.assertEqual((within_tol.status, within_tol.accepted, within_tol.anomaly), ("UPDATED", True, False))

        worse = finetune_trainer.judge_model_update(2.50, 2.55, cfg)
        self.assertEqual((worse.status, worse.accepted, worse.anomaly), ("KEPT_OLD", False, False))

        ceiling = finetune_trainer.judge_model_update(2.50, 3.08, cfg)
        self.assertEqual((ceiling.status, ceiling.accepted, ceiling.anomaly), ("ANOMALY", False, True))

        spike = finetune_trainer.judge_model_update(2.35, 2.75, cfg)
        self.assertEqual((spike.status, spike.accepted, spike.anomaly), ("ANOMALY", False, True))

        nan_case = finetune_trainer.judge_model_update(2.45, float("nan"), cfg)
        self.assertEqual((nan_case.status, nan_case.accepted, nan_case.anomaly), ("ANOMALY", False, True))

    def test_champion_vs_challenger_loads_clean_main_and_separates_best_val(self):
        loaded_revisions = []
        uploaded_branches = []

        def fake_load_candidates(candidates, device):
            first_repo, first_rev, _ = candidates[0]
            loaded_revisions.append((first_repo, first_rev))
            return _DummyModel(first_rev or "fallback")

        eval_returns = iter([2.48, 2.56, 2.53])

        def fake_eval(model, tokenizer, val_loader, device, ft_cfg=None):
            return next(eval_returns)

        def fake_train_epoch(model, tokenizer, train_loader, optimizer, scheduler_and_clip):
            model._state["epoch"] += 1
            return 2.40

        fake_tok = _DummyModel("tok")
        cfg = finetune_trainer.FineTuneConfig(symbol="XAUUSD", branch="xau-h1", epochs=2, push_to_hub=True)
        dummy_pair = (np.zeros((8, 10, 6), dtype=np.float32), np.zeros((8, 10, 5), dtype=np.float32))

        with (
            mock.patch.object(finetune_trainer, "_load_frozen_tokenizer", return_value=fake_tok),
            mock.patch.object(finetune_trainer, "_build_dataloaders", return_value=([1], [1])),
            mock.patch.object(finetune_trainer, "_load_kronos_from_candidates", side_effect=fake_load_candidates),
            mock.patch.object(finetune_trainer, "evaluate_validation_loss", side_effect=fake_eval),
            mock.patch.object(finetune_trainer, "_train_one_epoch", side_effect=fake_train_epoch),
            mock.patch.object(finetune_trainer, "_upload_finetuned_branch", side_effect=lambda *a: uploaded_branches.append(a[2])),
            mock.patch.object(finetune_trainer.torch, "optim", create=True) as fake_optim,
        ):
            fake_optim.AdamW.return_value = types.SimpleNamespace()
            fake_optim.lr_scheduler.OneCycleLR.return_value = types.SimpleNamespace()
            kept_metrics = finetune_trainer.run_specialist_finetune(
                dummy_pair, dummy_pair, cfg, repo_id="kingescaped/lvq-std-BOT", token="hf_write"
            )

        self.assertEqual(loaded_revisions, [("kingescaped/lvq-std-BOT", "xau-h1"), ("kingescaped/lvq-std-BOT", "main")])
        self.assertEqual(kept_metrics["baseline_val_loss"], 2.48)
        self.assertEqual(kept_metrics["best_val_loss"], 2.53)
        self.assertEqual(kept_metrics["status"], "KEPT_OLD")
        self.assertFalse(kept_metrics["improved"])
        self.assertFalse(kept_metrics["pushed"])
        self.assertEqual(uploaded_branches, [])
        self.assertIn("GIỮ BẢN CŨ", finetune_trainer.format_finetune_status_vn(kept_metrics))

    def test_upload_guard_blocks_overwriting_clean_main_branch(self):
        with self.assertRaises(ValueError):
            finetune_trainer._upload_finetuned_branch(_DummyModel("m"), "kingescaped/lvq-std-BOT", "main", "tok", {})


if __name__ == "__main__":
    unittest.main()
