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

import chart_visualizer
import config
import history_tracker
import kronos_predictor
import main
import telegram_notifier

VN_TZ = timezone(timedelta(hours=7))


def make_bars(count=45, base_price=70000.0, step_price=40.0, spread=80.0):
    t0 = datetime(2026, 11, 13, 18, 0, tzinfo=timezone.utc)
    times = [(t0 + timedelta(minutes=15 * i)).isoformat() for i in range(count)]
    closes = [base_price + step_price * i for i in range(count)]
    return FakeDataFrame({
        "time": times,
        "open": [c - step_price * 0.5 for c in closes],
        "high": [c + spread for c in closes],
        "low": [c - spread for c in closes],
        "close": closes,
        "volume": [100.0 + i for i in range(count)],
    })


class FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {"ok": True}

    def json(self):
        return self._payload


class TestP1Lite(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.orig_hist = history_tracker.HISTORY_FILE
        history_tracker.HISTORY_FILE = os.path.join(self.tmp_dir.name, "signals_history.json")

    def tearDown(self):
        history_tracker.HISTORY_FILE = self.orig_hist
        self.tmp_dir.cleanup()

    def test_config_has_p1_lite_settings_and_xau_30d(self):
        self.assertGreaterEqual(getattr(config, "INDICATOR_CANDLE_LIMIT", 0), 400)
        self.assertEqual(getattr(config, "KRONOS_SAMPLE_COUNT", None), 5)
        self.assertEqual(getattr(config, "KRONOS_TEMPERATURE", None), 0.6)
        self.assertTrue(bool(getattr(config, "STRATEGY_VERSION", "")))
        btc = next(s for s in config.SYMBOLS if s["name"] == "BTCUSD")
        xau = next(s for s in config.SYMBOLS if s["name"] == "XAUUSD")
        self.assertEqual(btc.get("digits"), 2)
        self.assertEqual(btc.get("threshold"), config.BTC_THRESHOLD)
        self.assertEqual(xau.get("period"), "30d")
        self.assertEqual(xau.get("digits"), 2)
        self.assertEqual(xau.get("threshold"), config.GOLD_THRESHOLD)

    def test_wilder_atr_and_adx_behavior(self):
        df = make_bars(count=40, step_price=0.0, spread=50.0)
        df._data["high"][-6] = df._data["close"][-6] + 500.0
        df._data["low"][-6] = df._data["close"][-6] - 500.0
        atr_val = kronos_predictor.calculate_volatility(df, window=14)
        self.assertLess(atr_val, 150.0)
        self.assertGreater(atr_val, 120.0)

        flat_df = make_bars(count=40, step_price=0.0, spread=50.0)
        self.assertLess(kronos_predictor.calculate_adx(flat_df), 20.0)

        trend_df = make_bars(count=40, step_price=40.0, spread=80.0)
        self.assertGreaterEqual(kronos_predictor.calculate_adx(trend_df), 20.0)

        short_df = make_bars(count=20, step_price=40.0, spread=80.0)
        self.assertEqual(kronos_predictor.calculate_adx(short_df), 0.0)

    def test_mc_agreement_confidence_consensus_vs_disagreement(self):
        df = make_bars(count=45, base_price=69000.0, step_price=40.0)
        ref = df.iloc[-1]["close"]

        def make_samples(pcts):
            dfs = [FakeDataFrame({"close": [ref * (1 + p / 100)], "high": [ref * (1 + p / 100) + 50], "low": [ref * (1 + p / 100) - 50]}) for p in pcts]
            mean_pct = sum(pcts) / len(pcts)
            mean_df = FakeDataFrame({"close": [ref * (1 + mean_pct / 100)], "high": [ref * 1.012], "low": [ref * 0.999]})
            return mean_df, dfs

        agree_pred = types.SimpleNamespace(predict=lambda **kw: make_samples([0.8, 0.9, 0.95, 1.0, 1.1]))
        with mock.patch.object(kronos_predictor, "load_kronos_base", return_value=agree_pred):
            res_ok = kronos_predictor.predict_future(df, "BTCUSD", timeframe="15m", pred_len=12)
        self.assertEqual(res_ok["sample_agreement"], 100.0)
        self.assertGreaterEqual(res_ok["confidence"], 75.0)
        self.assertGreater(res_ok["pred_end_price"], ref)

        mixed_pred = types.SimpleNamespace(predict=lambda **kw: make_samples([2.2, 2.2, 2.2, -1.2, -1.2]))
        with mock.patch.object(kronos_predictor, "load_kronos_base", return_value=mixed_pred):
            res_mix = kronos_predictor.predict_future(df, "BTCUSD", timeframe="15m", pred_len=12)
        self.assertEqual(res_mix["sample_agreement"], 60.0)
        self.assertLess(res_mix["confidence"], 75.0)

    def test_history_persists_metadata_and_keeps_legacy_compat(self):
        meta = {
            "confidence": 86.5, "pct_change": 0.95, "pred_end_price": 70665.0,
            "pred_high": 70800.0, "pred_low": 69900.0, "ref_close": 70000.0,
            "atr": 160.0, "adx": 31.2, "ema_50": 69500.0, "ema_200": 68800.0,
            "sample_agreement": 100.0, "sample_std_pct": 0.12, "strategy_version": "v3.2-p1-lite",
        }
        ok, _ = history_tracker.add_new_signal(
            "BTCUSD", "🟢 BUY (TĂNG)", 70010.0, 70800.0, 69770.0, "2026-11-14 07:00:00+00:00", metadata=meta,
        )
        self.assertTrue(ok)
        saved = history_tracker.load_history()[0]
        for k, v in meta.items():
            self.assertEqual(saved[k], v)

        history = history_tracker.load_history()
        yesterday = (datetime.now(VN_TZ) - timedelta(days=1)).strftime("%Y-%m-%d")
        history.append({
            "id": "XAUUSD_LEGACY", "symbol": "XAUUSD", "timeframe": "1h", "direction": "🔴 SELL (GIẢM)",
            "entry_price": 4100.0, "tp": 4080.0, "sl": 4115.0,
            "open_time": f"{yesterday} 10:00:00+00:00", "open_time_vn": f"{yesterday} 17:00:00",
            "status": "HIT_TP", "closed_at": f"{yesterday} 12:00:00+00:00", "close_price": 4080.0,
        })
        history_tracker.save_history(history)
        daily = history_tracker.generate_daily_report(target_date=yesterday)
        weekly = history_tracker.generate_weekly_report(ref_date=yesterday)
        self.assertIn("<b>", daily)
        self.assertIn("<b>", weekly)

    def test_main_fetches_indicator_limit_and_passes_metadata(self):
        df = make_bars(count=35, step_price=40.0)
        fake_res = {
            "current_price": 71300.0, "direction": "🟢 BUY (TĂNG)", "confidence": 85.0,
            "tp": 72000.0, "sl": 71000.0, "atr": 160.0, "ai_predicted": True, "pct_change": 1.1,
            "adx": 30.0, "ema_50": 70500.0, "ema_200": 69800.0, "sample_agreement": 100.0,
        }
        with (
            mock.patch.object(main, "check_and_send_daily_report_if_due"),
            mock.patch.object(main, "is_market_open", side_effect=lambda sym, _: (sym == "BTCUSD", "ok")),
            mock.patch.object(main, "get_candles", return_value=df) as get_spy,
            mock.patch.object(main, "update_open_signals"),
            mock.patch.object(main, "split_closed_and_live", return_value=(df, 71300.0, "t", True)),
            mock.patch.object(main, "predict_future", return_value=fake_res),
            mock.patch.object(main, "create_signal_chart", return_value=None),
            mock.patch.object(main, "add_new_signal", return_value=(True, "ID_1")) as add_spy,
            mock.patch.object(main, "send_telegram_message") as msg_spy,
        ):
            main.run_single_cycle(manual_force=False, now_vn=datetime(2026, 11, 14, 14, 2, tzinfo=VN_TZ))
        self.assertEqual(get_spy.call_args.kwargs["limit"], config.INDICATOR_CANDLE_LIMIT + 1)
        self.assertEqual(add_spy.call_args.kwargs["metadata"], fake_res)
        self.assertIn("<b>", msg_spy.call_args.args[0])

    def test_telegram_token_guard_and_message_retry(self):
        with mock.patch.object(config, "TELEGRAM_BOT_TOKEN", ""), mock.patch.object(telegram_notifier.requests, "post") as p_spy:
            self.assertFalse(telegram_notifier.send_telegram_message("hi"))
            p_spy.assert_not_called()

        calls = []
        def post_side_effect(*a, **kw):
            calls.append(kw)
            if len(calls) == 1:
                raise telegram_notifier.requests.RequestException("timeout")
            if len(calls) == 2:
                return FakeResp(429, {"ok": False, "description": "Too Many Requests", "parameters": {"retry_after": 0}})
            return FakeResp(200, {"ok": True})

        with (
            mock.patch.object(config, "TELEGRAM_BOT_TOKEN", "tok"),
            mock.patch.object(config, "TELEGRAM_GROUP_ID", "-1001"),
            mock.patch.object(telegram_notifier.time, "sleep"),
            mock.patch.object(telegram_notifier.requests, "post", side_effect=post_side_effect),
        ):
            self.assertTrue(telegram_notifier.send_telegram_message("<b>ok</b>"))
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["json"]["parse_mode"], "HTML")

    def test_telegram_photo_retry_and_4xx_no_retry(self):
        photo_path = os.path.join(self.tmp_dir.name, "chart.png")
        with open(photo_path, "wb") as f:
            f.write(b"PNG")
        p_calls = []
        def photo_effect(*a, **kw):
            p_calls.append(kw)
            return FakeResp(500, {"ok": False}) if len(p_calls) == 1 else FakeResp(200, {"ok": True})

        with (
            mock.patch.object(config, "TELEGRAM_BOT_TOKEN", "tok"),
            mock.patch.object(config, "TELEGRAM_GROUP_ID", "-1001"),
            mock.patch.object(telegram_notifier.time, "sleep"),
            mock.patch.object(telegram_notifier.requests, "post", side_effect=photo_effect),
        ):
            self.assertTrue(telegram_notifier.send_telegram_photo(photo_path, "<b>chart</b>"))
        self.assertEqual(len(p_calls), 2)
        self.assertEqual(p_calls[0]["data"]["parse_mode"], "HTML")

        with (
            mock.patch.object(config, "TELEGRAM_BOT_TOKEN", "tok"),
            mock.patch.object(config, "TELEGRAM_GROUP_ID", "-1001"),
            mock.patch.object(telegram_notifier.requests, "post", return_value=FakeResp(400, {"ok": False, "description": "bad"})) as bad_spy,
        ):
            self.assertFalse(telegram_notifier.send_telegram_message("bad"))
        self.assertEqual(bad_spy.call_count, 1)

    def test_chart_visualizer_closes_figures(self):
        df = make_bars(count=35)
        out = os.path.join(self.tmp_dir.name, "c.png")
        with (
            mock.patch.object(chart_visualizer.mpf, "plot"),
            mock.patch.object(chart_visualizer.plt, "close") as close_spy,
        ):
            chart_visualizer.create_signal_chart(df, "BTCUSD", "BTC", "15m", "$", "BUY", 70000.0, 71000.0, 69500.0, output_path=out)
        close_spy.assert_called_with("all")

        with (
            mock.patch.object(chart_visualizer.mpf, "plot", side_effect=RuntimeError("boom")),
            mock.patch.object(chart_visualizer.plt, "close") as close_err_spy,
        ):
            res = chart_visualizer.create_signal_chart(df, "BTCUSD", "BTC", "15m", "$", "BUY", 70000.0, 71000.0, 69500.0, output_path=out)
        self.assertIsNone(res)
        close_err_spy.assert_called_with("all")


if __name__ == "__main__":
    unittest.main()
