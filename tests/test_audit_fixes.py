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
import chart_visualizer
import config
import fetch_data
import history_tracker
import kronos_predictor
import main
import market_time
import telegram_notifier

VN_TZ = market_time.VN_TZ


def make_bars(count=45, base_price=69000.0, step_price=40.0, spread=80.0, tz=timezone.utc, step_min=15):
    t0 = datetime(2026, 11, 13, 18, 0, tzinfo=tz)
    times = [(t0 + timedelta(minutes=step_min * i)).isoformat() for i in range(count)]
    closes = [base_price + step_price * i for i in range(count)]
    return FakeDataFrame({
        "time": times,
        "open": [c - step_price * 0.5 for c in closes],
        "high": [c + spread for c in closes],
        "low": [c - spread for c in closes],
        "close": closes,
        "volume": [100.0 + i for i in range(count)],
    })


class FakeHttpResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"ok": True}

    def json(self):
        return self._payload


class TestAuditFixes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.orig_hist = history_tracker.HISTORY_FILE
        self.orig_state = bot_state.STATE_FILE
        history_tracker.HISTORY_FILE = os.path.join(self.tmp_dir.name, "signals_history.json")
        bot_state.STATE_FILE = os.path.join(self.tmp_dir.name, "bot_state.json")

    def tearDown(self):
        history_tracker.HISTORY_FILE = self.orig_hist
        bot_state.STATE_FILE = self.orig_state
        self.tmp_dir.cleanup()

    def test_mc_confidence_rejects_flat_plus_outlier_and_wild_dispersion(self):
        df = make_bars(count=45, base_price=69000.0, step_price=40.0)
        ref = df.iloc[-1]["close"]

        def build_samples(pcts):
            dfs = [
                FakeDataFrame({
                    "close": [ref * (1 + p / 100.0)],
                    "high": [ref * (1 + max(p, 0.0) / 100.0) + 400.0],
                    "low": [ref * (1 + min(p, 0.0) / 100.0) - 80.0],
                })
                for p in pcts
            ]
            mean_pct = sum(pcts) / len(pcts)
            mean_df = FakeDataFrame({
                "close": [ref * (1 + mean_pct / 100.0)],
                "high": [ref * 1.02],
                "low": [ref * 0.998],
            })
            return mean_df, dfs

        outlier_pred = types.SimpleNamespace(predict=lambda **kw: build_samples([0.01, 0.01, 0.01, 0.01, 2.80]))
        with mock.patch.object(kronos_predictor, "load_kronos_base", return_value=outlier_pred):
            res_out = kronos_predictor.predict_future(df, "BTCUSD", timeframe="15m", pred_len=12)
        self.assertIn("SIDEWAY", res_out["direction"])
        self.assertLess(res_out["confidence"], config.CONFIDENCE_THRESHOLD)

        dispersed_pred = types.SimpleNamespace(predict=lambda **kw: build_samples([0.30, 0.30, 0.60, 4.0, 6.0]))
        with mock.patch.object(kronos_predictor, "load_kronos_base", return_value=dispersed_pred):
            res_disp = kronos_predictor.predict_future(df, "BTCUSD", timeframe="15m", pred_len=12)
        self.assertLess(res_disp["confidence"], config.CONFIDENCE_THRESHOLD)

        four_of_five = types.SimpleNamespace(predict=lambda **kw: build_samples([0.75, 0.85, 0.90, 0.95, -0.10]))
        with mock.patch.object(kronos_predictor, "load_kronos_base", return_value=four_of_five):
            res_ok = kronos_predictor.predict_future(df, "BTCUSD", timeframe="15m", pred_len=12)
        self.assertIn("BUY", res_ok["direction"])
        self.assertGreaterEqual(res_ok["confidence"], config.CONFIDENCE_THRESHOLD)

    def test_real_rr_filter_and_directional_slippage(self):
        df = make_bars(count=45, base_price=69000.0, step_price=40.0, spread=80.0)
        ref = df.iloc[-1]["close"]

        low_rr_pred = types.SimpleNamespace(
            predict=lambda **kw: FakeDataFrame({"close": [ref * 1.006], "high": [ref + 120.0], "low": [ref - 50.0]})
        )
        with mock.patch.object(kronos_predictor, "load_kronos_base", return_value=low_rr_pred):
            res_low_rr = kronos_predictor.predict_future(df, "BTCUSD", timeframe="15m", pred_len=12, live_price=ref)
        self.assertIn("SIDEWAY", res_low_rr["direction"])
        self.assertEqual(res_low_rr["tp"], round(ref, 2))

        good_pred = types.SimpleNamespace(
            predict=lambda **kw: FakeDataFrame({"close": [ref * 1.012], "high": [ref + 900.0], "low": [ref - 50.0]})
        )
        with mock.patch.object(kronos_predictor, "load_kronos_base", return_value=good_pred):
            res_pullback = kronos_predictor.predict_future(df, "BTCUSD", timeframe="15m", pred_len=12, live_price=ref - 65.0)
            res_deep_dump = kronos_predictor.predict_future(df, "BTCUSD", timeframe="15m", pred_len=12, live_price=ref - 200.0)

        self.assertIn("BUY", res_pullback["direction"])
        self.assertFalse(res_pullback["slippage_blocked"])
        self.assertEqual(res_pullback["tp"], round(ref + 900.0, 2))
        self.assertTrue(res_deep_dump["slippage_blocked"])
        self.assertIn("SIDEWAY", res_deep_dump["direction"])

    def test_telegram_failure_rolls_back_signal_and_photo_falls_back_to_text(self):
        item = config.SYMBOLS[0]
        res = {
            "current_price": 70000.0, "direction": "🟢 BUY (TĂNG)", "confidence": 85.0,
            "tp": 70800.0, "sl": 69500.0, "atr": 300.0, "ai_predicted": True, "pct_change": 1.1,
        }
        with (
            mock.patch.object(main, "create_signal_chart", return_value=None),
            mock.patch.object(main, "send_telegram_message", return_value=False),
        ):
            main._handle_symbol_result(item, FakeDataFrame({}), "2026-11-27T14:00:00+07:00", res, manual_force=False)
        self.assertEqual(history_tracker.load_history(), [])

        photo_path = os.path.join(self.tmp_dir.name, "chart.png")
        with open(photo_path, "wb") as f:
            f.write(b"PNG")

        def post_effect(url, **kw):
            if url.endswith("/sendPhoto"):
                return FakeHttpResp(400, {"ok": False, "description": "Bad image"})
            return FakeHttpResp(200, {"ok": True})

        with (
            mock.patch.object(config, "TELEGRAM_BOT_TOKEN", "tok"),
            mock.patch.object(config, "TELEGRAM_GROUP_ID", "-1001"),
            mock.patch.object(telegram_notifier.requests, "post", side_effect=post_effect) as post_spy,
        ):
            self.assertTrue(telegram_notifier.send_telegram_photo(photo_path, "<b>caption</b>"))
        self.assertEqual(post_spy.call_count, 2)

    def test_stale_chart_removed_on_error_and_force_always_dispatches(self):
        df = make_bars(count=35)
        stale_path = os.path.join(self.tmp_dir.name, "stale.png")
        with open(stale_path, "wb") as f:
            f.write(b"OLD_PNG")
        captured_title = {}
        with mock.patch.object(chart_visualizer.mpf, "plot", side_effect=lambda *a, **kw: captured_title.update(kw.get("title", {}))):
            chart_visualizer.create_signal_chart(df, "BTCUSD", "BTC", "15m", "$", "BUY", 70000.0, 70800.0, 69500.0, output_path=stale_path)
        self.assertIn("LVQ Std SIGNAL", captured_title.get("title", ""))

        with open(stale_path, "wb") as f:
            f.write(b"OLD_PNG")
        with mock.patch.object(chart_visualizer.mpf, "plot", side_effect=RuntimeError("plot fail")):
            out = chart_visualizer.create_signal_chart(df, "BTCUSD", "BTC", "15m", "$", "BUY", 70000.0, 70800.0, 69500.0, output_path=stale_path)
        self.assertIsNone(out)
        self.assertFalse(os.path.exists(stale_path))

        item = config.SYMBOLS[0]
        res = {
            "current_price": 70000.0, "direction": "🟢 BUY (TĂNG)", "confidence": 86.0,
            "tp": 70800.0, "sl": 69500.0, "atr": 300.0, "ai_predicted": True, "pct_change": 1.1,
        }
        history_tracker.add_new_signal("BTCUSD", "🟢 BUY (TĂNG)", 70000.0, 70800.0, 69500.0, "2026-11-27T14:00:00+07:00")
        off_h1_res = dict(res, _off_schedule_h1=True)
        with (
            mock.patch.object(main, "create_signal_chart", return_value=None),
            mock.patch.object(main, "send_telegram_message", return_value=True) as msg_spy,
        ):
            main._handle_symbol_result(item, df, "2026-11-27T14:00:00+07:00", res, manual_force=True)
            main._handle_symbol_result(config.SYMBOLS[1], df, "2026-11-27T14:00:00+07:00", off_h1_res, manual_force=True)
        self.assertEqual(msg_spy.call_count, 2)
        self.assertEqual(len(history_tracker.load_history()), 1)

    def test_two_order_dca_kept_and_labeled_in_caption(self):
        ok1, id1 = history_tracker.add_new_signal("BTCUSD", "🟢 BUY (TĂNG)", 70000.0, 70900.0, 69550.0, "2026-11-27T14:00:00+07:00", atr=300.0)
        ok2, id2 = history_tracker.add_new_signal("BTCUSD", "🟢 BUY (TĂNG)", 69700.0, 70600.0, 69250.0, "2026-11-27T14:15:00+07:00", atr=300.0)
        ok3, _ = history_tracker.add_new_signal("BTCUSD", "🟢 BUY (TĂNG)", 69400.0, 70300.0, 68950.0, "2026-11-27T14:30:00+07:00", atr=300.0)
        self.assertTrue(ok1)
        self.assertTrue(ok2)
        self.assertFalse(ok3)
        self.assertNotEqual(id1, id2)
        hist = history_tracker.load_history()
        self.assertFalse(hist[0]["is_dca"])
        self.assertTrue(hist[1]["is_dca"])
        self.assertEqual(hist[1]["order_num"], 2)

        cap_dca = main._build_signal_caption(
            config.SYMBOLS[0],
            {"current_price": 69700.0, "direction": "🟢 BUY (TĂNG)", "confidence": 84.0, "pct_change": 1.0, "tp": 70600.0, "sl": 69250.0, "atr": 300.0, "forecast_window_vn": "14:15 → 17:15 VN"},
            "M15",
            order_num=2,
        )
        self.assertIn("DCA", cap_dca)
        self.assertIn("14:15 → 17:15 VN", cap_dca)

    def test_vietnam_timezone_us_dst_and_future_timestamps_skip_closed_hours(self):
        ny_tz = timezone(timedelta(hours=-5))

        class ResettableFakeDF(FakeDataFrame):
            def __getitem__(self, key):
                if isinstance(key, list):
                    return ResettableFakeDF({k: self._data[k] for k in key})
                return super().__getitem__(key)

            def tail(self, n):
                return ResettableFakeDF({k: v[-n:] for k, v in self._data.items()})

            def copy(self):
                return ResettableFakeDF(self._data)

            def reset_index(self, inplace=False):
                self._data = {"Date": [datetime(2026, 11, 27, 2, 0, tzinfo=ny_tz)], **self._data}
                self.columns = list(self._data.keys())

        raw_df = ResettableFakeDF({
            "Open": [4100.0], "High": [4110.0], "Low": [4095.0], "Close": [4105.0], "Volume": [500.0],
        })
        with mock.patch.object(fetch_data.yf, "download", return_value=raw_df):
            fetched = fetch_data.get_candles("GC=F", timeframe="1h", period="5d", limit=10)
        self.assertEqual(fetched["time"].iloc[-1].utcoffset(), timedelta(hours=7))
        self.assertEqual(fetched["time"].iloc[-1].hour, 14)

        winter_break = datetime(2026, 11, 25, 5, 30, tzinfo=VN_TZ)
        winter_open = datetime(2026, 11, 25, 4, 30, tzinfo=VN_TZ)
        summer_break = datetime(2026, 7, 15, 4, 30, tzinfo=VN_TZ)
        summer_open = datetime(2026, 7, 15, 5, 30, tzinfo=VN_TZ)
        self.assertFalse(market_time.is_market_open("XAUUSD", winter_break)[0])
        self.assertTrue(market_time.is_market_open("XAUUSD", winter_open)[0])
        self.assertFalse(market_time.is_market_open("XAUUSD", summer_break)[0])
        self.assertTrue(market_time.is_market_open("XAUUSD", summer_open)[0])

        fri_close_bar = "2026-11-27T16:00:00-05:00"
        fut = market_time.build_future_timestamps(fri_close_bar, "XAUUSD", "1h", 6)
        self.assertEqual(len(fut), 6)
        self.assertEqual(market_time.format_forecast_window_vn(fut, "1h"), "06:00 → 12:00 VN")

    def test_1m_refresh_catches_post_entry_wick_and_report_runs_after_updates(self):
        history_tracker.save_history([{
            "id": "BTCUSD_20261127_140210", "symbol": "BTCUSD", "timeframe": "15m",
            "direction": "🟢 BUY (TĂNG)", "entry_price": 70000.0, "tp": 70500.0, "sl": 69700.0,
            "open_time": "2026-11-27T13:45:00+07:00", "open_time_vn": "2026-11-27 14:02:10",
            "status": "OPEN", "closed_at": None, "close_price": None,
        }])
        bars_1m = FakeDataFrame({
            "time": ["2026-11-27T14:02:00+07:00", "2026-11-27T14:05:00+07:00"],
            "open": [70000.0, 69900.0], "high": [70600.0, 69950.0],
            "low": [69400.0, 69650.0], "close": [70000.0, 69800.0], "volume": [10.0, 12.0],
        })
        bars_15m = make_bars(count=35, base_price=70000.0, step_price=0.0, spread=20.0, tz=VN_TZ)

        def candle_router(ticker, timeframe="15m", period="5d", limit=200):
            return bars_1m if timeframe == "1m" else bars_15m

        call_order = []
        with (
            mock.patch.object(main, "is_market_open", side_effect=lambda sym, _: (sym == "BTCUSD", "ok")),
            mock.patch.object(main, "get_candles", side_effect=candle_router),
            mock.patch.object(main, "predict_future", return_value={"current_price": 70000.0, "direction": "⚪ SIDEWAY", "confidence": 50.0, "tp": 70000.0, "sl": 70000.0, "ai_predicted": True}),
            mock.patch.object(main, "check_and_send_daily_report_if_due", side_effect=lambda **kw: call_order.append(("report", history_tracker.load_history()[0]["status"]))),
        ):
            main.run_single_cycle(manual_force=False, now_vn=datetime(2026, 11, 27, 14, 15, tzinfo=VN_TZ))

        self.assertEqual(history_tracker.load_history()[0]["status"], "HIT_SL")
        self.assertEqual(call_order, [("report", "HIT_SL")])

    def test_dotenv_cli_exit_code_carryover_report_and_weekly_mon_fri(self):
        history_tracker.save_history([{
            "id": "XAUUSD_20261125_233000", "symbol": "XAUUSD", "timeframe": "1h",
            "direction": "🟢 BUY (TĂNG)", "entry_price": 4100.0, "tp": 4130.0, "sl": 4085.0,
            "open_time": "2026-11-25T23:00:00+07:00", "open_time_vn": "2026-11-25 23:30:00",
            "status": "HIT_TP", "closed_at": "2026-11-26T08:00:00+07:00", "close_price": 4130.0,
        }, {
            "id": "BTCUSD_20261128_100000", "symbol": "BTCUSD", "timeframe": "15m",
            "direction": "🟢 BUY (TĂNG)", "entry_price": 70000.0, "tp": 70600.0, "sl": 69600.0,
            "open_time": "2026-11-28T10:00:00+07:00", "open_time_vn": "2026-11-28 10:00:00",
            "status": "HIT_TP", "closed_at": "2026-11-28T11:00:00+07:00", "close_price": 70600.0,
        }])
        daily_26 = history_tracker.generate_daily_report(target_date="2026-11-26")
        weekly_sat = history_tracker.generate_weekly_report(ref_date="2026-11-28")
        self.assertIn("Lệnh tồn ngày trước chốt hôm qua", daily_26)
        self.assertIn("<code>1</code>", weekly_sat)

        env_file = os.path.join(self.tmp_dir.name, ".env")
        with open(env_file, "w", encoding="utf-8-sig") as f:
            f.write('TEST_DOTENV_NEW="hello_vn"\nTEST_DOTENV_KEEP=override_me\n')
        with mock.patch.dict(os.environ, {"TEST_DOTENV_KEEP": "existing"}, clear=False):
            os.environ.pop("TEST_DOTENV_NEW", None)
            config._load_local_dotenv(env_file)
            self.assertEqual(os.environ.get("TEST_DOTENV_NEW"), "hello_vn")
            self.assertEqual(os.environ.get("TEST_DOTENV_KEEP"), "existing")
            os.environ.pop("TEST_DOTENV_NEW", None)

        class FakeDTIndex(list):
            pass

        with (
            mock.patch.object(kronos_predictor.pd, "DatetimeIndex", FakeDTIndex),
            mock.patch.object(kronos_predictor.pd, "to_datetime", side_effect=lambda vals: FakeDTIndex([datetime.fromisoformat(v) for v in vals])),
            mock.patch.object(kronos_predictor.pd, "Series", side_effect=lambda idx, index=None: types.SimpleNamespace(iloc=idx)),
        ):
            df_idx = make_bars(count=30)
            _, x_ts, y_ts, _ = kronos_predictor._prepare_kronos_inputs(df_idx, kronos_predictor._get_symbol_spec("BTCUSD", "15m"), 6)
            self.assertTrue(hasattr(x_ts, "iloc"))
            self.assertIsInstance(y_ts, FakeDTIndex)

        with (
            mock.patch.object(sys, "argv", ["main.py", "--test"]),
            mock.patch.object(main, "test_telegram", return_value=False),
        ):
            with self.assertRaises(SystemExit) as ctx:
                main.main()
            self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
