import ast
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

import config
import history_tracker
import kronos_predictor
import main
from model import kronos as model_kronos

VN_TZ = timezone(timedelta(hours=7))


class FakeModelObj:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        return self


def make_candle_df(n=40, start_price=70000.0, step=100.0, start_utc=None):
    base = start_utc or datetime(2026, 11, 14, 0, 0, tzinfo=timezone.utc)
    times = [(base + timedelta(minutes=15 * i)).isoformat() for i in range(n)]
    closes = [start_price + step * i for i in range(n)]
    return FakeDataFrame({
        "time": times,
        "open": [c - 20.0 for c in closes],
        "high": [c + 80.0 for c in closes],
        "low": [c - 80.0 for c in closes],
        "close": closes,
        "volume": [100.0 + i for i in range(n)],
    })


class TestP0Fixes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.hist_path = os.path.join(self.tmp_dir.name, "signals_history.json")
        self.state_path = os.path.join(self.tmp_dir.name, "bot_state.json")
        self.orig_hist = history_tracker.HISTORY_FILE
        history_tracker.HISTORY_FILE = self.hist_path

    def tearDown(self):
        history_tracker.HISTORY_FILE = self.orig_hist
        self.tmp_dir.cleanup()

    def test_load_kronos_calls_eval_on_model_and_tokenizer(self):
        tok_obj = FakeModelObj()
        mod_obj = FakeModelObj()
        fake_model_pkg = types.ModuleType("model")
        fake_model_pkg.KronosTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: tok_obj)
        fake_model_pkg.Kronos = types.SimpleNamespace(from_pretrained=lambda *a, **k: mod_obj)
        fake_model_pkg.KronosPredictor = lambda model, tokenizer, **k: types.SimpleNamespace(model=model, tokenizer=tokenizer)

        kronos_predictor._GLOBAL_PREDICTOR = None
        kronos_predictor._MODEL_LOADED = False
        with mock.patch.dict(sys.modules, {"model": fake_model_pkg}):
            pred = kronos_predictor.load_kronos_base()
        self.assertIsNotNone(pred)
        self.assertTrue(tok_obj.eval_called)
        self.assertTrue(mod_obj.eval_called)

    def test_kronos_predictor_init_calls_eval_and_uses_inference_mode(self):
        tok_obj = FakeModelObj()
        mod_obj = FakeModelObj()
        model_kronos.KronosPredictor(model=mod_obj, tokenizer=tok_obj, device="cpu")
        self.assertTrue(tok_obj.eval_called)
        self.assertTrue(mod_obj.eval_called)

        kronos_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "kronos.py")
        with open(kronos_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "auto_regressive_inference")
        attrs = [n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)]
        self.assertIn("inference_mode", attrs)
        self.assertNotIn("no_grad", attrs)

    def test_fallback_disabled_for_signals_when_ai_unavailable(self):
        df = make_candle_df(n=40, start_price=70000.0, step=150.0)
        with mock.patch.object(kronos_predictor, "load_kronos_base", return_value=None):
            res = kronos_predictor.predict_future(df, "BTCUSD", timeframe="15m", pred_len=12)
        self.assertFalse(res["ai_predicted"])
        self.assertEqual(res["confidence"], 0.0)
        self.assertIn("SIDEWAY", res["direction"])
        self.assertEqual(res["tp"], res["current_price"])
        self.assertEqual(res["sl"], res["current_price"])

    def test_predict_future_cleans_nan_and_keeps_live_candle(self):
        df = make_candle_df(n=40, start_price=70000.0, step=50.0)
        df._data["close"][10] = float("nan")
        df._data["volume"][-1] = float("nan")
        captured = {}

        def fake_predict(df, x_timestamp, y_timestamp, pred_len, **kwargs):
            captured["x_df"] = df
            captured["last_x_time"] = x_timestamp.iloc[-1]
            return FakeDataFrame({"close": [73000.0], "high": [73100.0], "low": [72900.0]})

        fake_pred = types.SimpleNamespace(predict=fake_predict)
        with mock.patch.object(kronos_predictor, "load_kronos_base", return_value=fake_pred):
            res = kronos_predictor.predict_future(df, "BTCUSD", timeframe="15m", pred_len=12)
        self.assertTrue(res["ai_predicted"])
        self.assertEqual(len(captured["x_df"]), 39)
        self.assertEqual(captured["last_x_time"].isoformat(), df._data["time"][-1])

    def test_main_blocks_non_ai_signal_even_if_high_confidence(self):
        fake_df = make_candle_df(n=30)
        fake_res = {
            "current_price": 70000.0, "direction": "🟢 BUY (TĂNG)", "confidence": 85.0,
            "tp": 71000.0, "sl": 69500.0, "atr": 300.0, "ai_predicted": False, "pct_change": 1.4,
        }
        with (
            mock.patch.object(main, "check_and_send_daily_report_if_due"),
            mock.patch.object(main, "is_market_open", return_value=(True, "ok")),
            mock.patch.object(main, "get_candles", return_value=fake_df),
            mock.patch.object(main, "update_open_signals"),
            mock.patch.object(main, "predict_future", return_value=fake_res),
            mock.patch.object(main, "create_signal_chart", return_value=None),
            mock.patch.object(main, "add_new_signal") as add_spy,
            mock.patch.object(main, "send_telegram_message") as msg_spy,
            mock.patch.object(main, "send_telegram_photo") as photo_spy,
        ):
            main.run_single_cycle(manual_force=False, now_vn=datetime(2026, 11, 14, 14, 2, tzinfo=VN_TZ))
        add_spy.assert_not_called()
        msg_spy.assert_not_called()
        photo_spy.assert_not_called()

    def test_update_open_signals_excludes_open_candle_and_is_sl_first(self):
        history_tracker.save_history([{
            "id": "BTCUSD_20261114_140212", "symbol": "BTCUSD", "timeframe": "15m",
            "direction": "🟢 BUY (TĂNG)", "entry_price": 70000.0, "tp": 70500.0, "sl": 69500.0,
            "open_time": "2026-11-14 07:00:00+00:00", "open_time_vn": "2026-11-14 14:02:12",
            "status": "OPEN", "closed_at": None, "close_price": None,
        }])
        same_bar = FakeDataFrame({
            "time": ["2026-11-14 07:00:00+00:00"], "high": [71000.0], "low": [69000.0], "close": [70000.0],
        })
        self.assertFalse(history_tracker.update_open_signals("BTCUSD", 71000.0, 69000.0, 70000.0, "2026-11-14 07:00:00+00:00", df=same_bar))
        self.assertEqual(history_tracker.load_history()[0]["status"], "OPEN")

        self.assertFalse(history_tracker.update_open_signals("BTCUSD", 71000.0, 69000.0, 70000.0, "2026-11-14 07:00:00+00:00", df=None))
        dual_bar = FakeDataFrame({
            "time": ["2026-11-14 07:00:00+00:00", "2026-11-14 07:15:00+00:00"],
            "high": [70100.0, 70800.0], "low": [69900.0, 69400.0], "close": [70000.0, 70100.0],
        })
        self.assertTrue(history_tracker.update_open_signals("BTCUSD", 70800.0, 69400.0, 70100.0, "2026-11-14 07:15:00+00:00", df=dual_bar))
        self.assertEqual(history_tracker.load_history()[0]["status"], "HIT_SL")

    def test_update_open_signals_mid_hour_15m_and_sell_sl_first_fallback(self):
        history_tracker.save_history([{
            "id": "XAUUSD_20261114_140212", "symbol": "XAUUSD", "timeframe": "1h",
            "direction": "🔴 SELL (GIẢM)", "entry_price": 4100.0, "tp": 4080.0, "sl": 4120.0,
            "open_time": "2026-11-14 06:00:00+00:00", "open_time_vn": "2026-11-14 14:02:12",
            "status": "OPEN", "closed_at": None, "close_price": None,
        }])
        m15_df = FakeDataFrame({
            "time": ["2026-11-14 06:30:00+00:00", "2026-11-14 07:00:00+00:00", "2026-11-14 07:15:00+00:00"],
            "high": [4150.0, 4150.0, 4105.0], "low": [4050.0, 4050.0, 4075.0], "close": [4100.0, 4100.0, 4080.0],
        })
        self.assertTrue(history_tracker.update_open_signals("XAUUSD", 4105.0, 4075.0, 4080.0, "2026-11-14 07:15:00+00:00", df=m15_df))
        rec = history_tracker.load_history()[0]
        self.assertEqual(rec["status"], "HIT_TP")
        self.assertEqual(rec["closed_at"], "2026-11-14 07:15:00+00:00")

        history_tracker.save_history([{
            "id": "XAUUSD_20261114_150212", "symbol": "XAUUSD", "timeframe": "1h",
            "direction": "🔴 SELL (GIẢM)", "entry_price": 4100.0, "tp": 4080.0, "sl": 4120.0,
            "open_time": "2026-11-14 07:00:00+00:00", "open_time_vn": "2026-11-14 15:02:12",
            "status": "OPEN", "closed_at": None, "close_price": None,
        }])
        self.assertTrue(history_tracker.update_open_signals("XAUUSD", 4125.0, 4075.0, 4100.0, "2026-11-14 08:15:00+00:00", df=None))
        self.assertEqual(history_tracker.load_history()[0]["status"], "HIT_SL")

    def test_atomic_save_and_corrupt_history_raises(self):
        history_tracker.save_history([{"id": "SIG_1"}])
        self.assertEqual(history_tracker.load_history(), [{"id": "SIG_1"}])
        leftovers = [f for f in os.listdir(self.tmp_dir.name) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])
        with open(self.hist_path, "w", encoding="utf-8") as f:
            f.write("{broken json")
        with self.assertRaises(ValueError):
            history_tracker.load_history()

    def test_hedging_blocked_duplicates_checked_and_dead_config_removed(self):
        self.assertFalse(hasattr(config, "ALLOW_MULTIPLE_SIGNALS"))
        self.assertIs(getattr(config, "ALLOW_OPPOSITE_HEDGING", None), False)

        ok, _ = history_tracker.add_new_signal("BTCUSD", "🟢 BUY (TĂNG)", 70000.0, 71000.0, 69500.0, "2026-11-14 07:00:00+00:00", "15m", atr=500.0)
        self.assertTrue(ok)
        ok_hedge, _ = history_tracker.add_new_signal("BTCUSD", "🔴 SELL (GIẢM)", 70100.0, 69000.0, 70600.0, "2026-11-14 07:15:00+00:00", "15m", atr=500.0)
        self.assertFalse(ok_hedge)

        ok_e2, _ = history_tracker.add_new_signal("BTCUSD", "🟢 BUY (TĂNG)", 69500.0, 70500.0, 69000.0, "2026-11-14 07:30:00+00:00", "15m", atr=500.0)
        self.assertTrue(ok_e2)
        self.assertEqual(history_tracker.load_history()[-1]["order_num"], 2)

        ok_e3, _ = history_tracker.add_new_signal("BTCUSD", "🟢 BUY (TĂNG)", 69000.0, 70000.0, 68500.0, "2026-11-14 07:45:00+00:00", "15m", atr=500.0)
        self.assertFalse(ok_e3)

        history = history_tracker.load_history()
        for i in range(12):
            history.append({"id": f"XAU_{i}", "symbol": "XAUUSD", "open_time": f"2026-11-15 0{i % 9}:00:00+00:00", "status": "HIT_TP"})
        history_tracker.save_history(history)
        ok_dup, _ = history_tracker.add_new_signal("BTCUSD", "🟢 BUY (TĂNG)", 69100.0, 70000.0, 68600.0, "2026-11-14 07:00:00+00:00", "15m", atr=500.0)
        self.assertFalse(ok_dup)

    def test_report_dedup_and_minute_15_guard(self):
        import bot_state
        with (
            mock.patch.object(bot_state, "STATE_FILE", self.state_path),
            mock.patch.object(main, "generate_daily_report", return_value="daily"),
            mock.patch.object(main, "generate_weekly_report", return_value="weekly"),
            mock.patch.object(main, "send_telegram_message", return_value=True) as send_spy,
        ):
            main.check_and_send_daily_report_if_due(now_vn=datetime(2026, 11, 21, 6, 2, tzinfo=VN_TZ))
            self.assertEqual(send_spy.call_count, 2)
            main.check_and_send_daily_report_if_due(now_vn=datetime(2026, 11, 21, 6, 5, tzinfo=VN_TZ))
            self.assertEqual(send_spy.call_count, 2)
            main.check_and_send_daily_report_if_due(now_vn=datetime(2026, 11, 22, 6, 25, tzinfo=VN_TZ))
            self.assertEqual(send_spy.call_count, 3)
            main.check_and_send_daily_report_if_due(now_vn=datetime(2026, 11, 23, 7, 0, tzinfo=VN_TZ))
            self.assertEqual(send_spy.call_count, 3)

        with (
            mock.patch.object(main, "check_and_send_daily_report_if_due"),
            mock.patch.object(main, "is_market_open", side_effect=lambda sym, _: (sym == "XAUUSD", "ok")),
            mock.patch.object(main, "get_candles", return_value=None) as get_spy,
        ):
            main.run_single_cycle(manual_force=False, now_vn=datetime(2026, 11, 20, 14, 15, tzinfo=VN_TZ))
        get_spy.assert_called_once_with("GC=F", timeframe="15m", period="2d", limit=24)


if __name__ == "__main__":
    unittest.main()
