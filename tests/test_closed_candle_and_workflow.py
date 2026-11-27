import os
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.stubs import FakeDataFrame, install_missing_stubs

install_missing_stubs()

import config
import fetch_data
import kronos_predictor
import main

VN_TZ = timezone(timedelta(hours=7))


def make_bars(count=35, start_utc=None, step_min=15, base_price=70000.0, step_price=0.0):
    t0 = start_utc or datetime(2026, 11, 13, 22, 30, tzinfo=timezone.utc)
    times = [(t0 + timedelta(minutes=step_min * i)).isoformat() for i in range(count)]
    closes = [base_price + step_price * i for i in range(count)]
    return FakeDataFrame({
        "time": times,
        "open": [c for c in closes],
        "high": [c + 80.0 for c in closes],
        "low": [c - 80.0 for c in closes],
        "close": closes,
        "volume": [100.0 + i for i in range(count)],
    })


class TestClosedCandleAndWorkflow(unittest.TestCase):
    def test_split_closed_and_live_drops_forming_btc_m15(self):
        start = datetime(2026, 11, 14, 0, 0, tzinfo=timezone.utc)
        df = make_bars(count=29, start_utc=start, step_min=15, base_price=70000.0)
        df._data["close"][-1] = 70050.0
        now_utc = datetime(2026, 11, 14, 7, 2, tzinfo=timezone.utc)

        closed_df, live_price, _, had_forming = fetch_data.split_closed_and_live(df, timeframe="15m", now_utc=now_utc)
        self.assertTrue(had_forming)
        self.assertEqual(len(closed_df), 28)
        self.assertEqual(str(closed_df.iloc[-1]["time"]), "2026-11-14T06:45:00+00:00")
        self.assertEqual(live_price, 70050.0)

    def test_split_closed_and_live_xau_h1_and_nan_live_fallback(self):
        start_h1 = datetime(2026, 11, 13, 2, 0, tzinfo=timezone.utc)
        df_h1 = make_bars(count=29, start_utc=start_h1, step_min=60, base_price=4000.0)
        now_utc = datetime(2026, 11, 14, 7, 2, tzinfo=timezone.utc)

        closed_h1, live_h1, _, had_forming_h1 = fetch_data.split_closed_and_live(df_h1, timeframe="1h", now_utc=now_utc)
        self.assertFalse(had_forming_h1)
        self.assertEqual(len(closed_h1), 29)
        self.assertEqual(live_h1, 4000.0)

        df_m15 = make_bars(count=29, start_utc=datetime(2026, 11, 14, 0, 0, tzinfo=timezone.utc), step_min=15, base_price=69900.0)
        df_m15._data["close"][-2] = 69980.0
        df_m15._data["close"][-1] = float("nan")
        closed_m15, live_m15, _, had_forming_m15 = fetch_data.split_closed_and_live(df_m15, timeframe="15m", now_utc=now_utc)
        self.assertTrue(had_forming_m15)
        self.assertEqual(live_m15, 69980.0)
        self.assertEqual(len(closed_m15), 28)

    def test_predict_future_live_price_within_slippage_reanchors_targets(self):
        df_buy = make_bars(count=35, base_price=69000.0, step_price=30.0)
        ref_close = df_buy.iloc[-1]["close"]
        fake_buy = types.SimpleNamespace(predict=lambda **kw: FakeDataFrame({"close": [ref_close + 1200.0], "high": [ref_close + 1500.0], "low": [ref_close - 100.0]}))
        with mock.patch.object(kronos_predictor, "load_kronos_base", return_value=fake_buy):
            res_buy = kronos_predictor.predict_future(df_buy, "BTCUSD", timeframe="15m", pred_len=12, live_price=ref_close + 30.0)
        self.assertTrue(res_buy["ai_predicted"])
        self.assertIn("BUY", res_buy["direction"])
        self.assertFalse(res_buy["slippage_blocked"])
        self.assertEqual(res_buy["current_price"], round(ref_close + 30.0, 2))
        self.assertEqual(res_buy["ref_close"], round(ref_close, 2))
        self.assertGreater(res_buy["tp"], res_buy["current_price"])
        self.assertLess(res_buy["sl"], res_buy["current_price"])
        rr_buy = (res_buy["tp"] - res_buy["current_price"]) / (res_buy["current_price"] - res_buy["sl"])
        self.assertGreaterEqual(rr_buy, config.MIN_RR_RATIO - 1e-3)

        df_sell = make_bars(count=35, base_price=71000.0, step_price=-30.0)
        sell_ref = df_sell.iloc[-1]["close"]
        fake_sell = types.SimpleNamespace(predict=lambda **kw: FakeDataFrame({"close": [sell_ref - 1200.0], "high": [sell_ref + 100.0], "low": [sell_ref - 1500.0]}))
        with mock.patch.object(kronos_predictor, "load_kronos_base", return_value=fake_sell):
            res_sell = kronos_predictor.predict_future(df_sell, "BTCUSD", timeframe="15m", pred_len=12, live_price=sell_ref - 30.0)
        self.assertIn("SELL", res_sell["direction"])
        self.assertFalse(res_sell["slippage_blocked"])
        self.assertLess(res_sell["tp"], res_sell["current_price"])
        self.assertGreater(res_sell["sl"], res_sell["current_price"])
        rr_sell = (res_sell["current_price"] - res_sell["tp"]) / (res_sell["sl"] - res_sell["current_price"])
        self.assertGreaterEqual(rr_sell, config.MIN_RR_RATIO - 1e-3)

    def test_predict_future_blocks_signal_when_slippage_exceeds_cap(self):
        df = make_bars(count=35, base_price=69000.0, step_price=30.0)
        ref_close = df.iloc[-1]["close"]
        fake_pred = types.SimpleNamespace(predict=lambda **kw: FakeDataFrame({"close": [ref_close + 1200.0], "high": [ref_close + 1500.0], "low": [ref_close - 100.0]}))
        with mock.patch.object(kronos_predictor, "load_kronos_base", return_value=fake_pred):
            res = kronos_predictor.predict_future(df, "BTCUSD", timeframe="15m", pred_len=12, live_price=ref_close + 200.0)
        self.assertTrue(res["slippage_blocked"])
        self.assertIn("SIDEWAY", res["direction"])
        self.assertLess(res["confidence"], config.CONFIDENCE_THRESHOLD)
        self.assertEqual(res["tp"], round(ref_close + 200.0, 2))
        self.assertEqual(res["sl"], round(ref_close + 200.0, 2))

    def test_main_passes_closed_df_and_live_price_to_predict_future(self):
        df = make_bars(count=29, start_utc=datetime(2026, 11, 14, 0, 0, tzinfo=timezone.utc), step_min=15, base_price=70000.0)
        df._data["close"][-1] = 70050.0
        captured = {}

        def fake_predict(closed_df, sym, timeframe="15m", pred_len=12, live_price=None):
            captured["last_closed_time"] = str(closed_df.iloc[-1]["time"])
            captured["live_price"] = live_price
            return {
                "current_price": 70050.0, "direction": "🟢 BUY (TĂNG)", "confidence": 82.0,
                "tp": 70800.0, "sl": 69700.0, "atr": 160.0, "ai_predicted": True, "pct_change": 1.1,
            }

        with (
            mock.patch.object(main, "check_and_send_daily_report_if_due"),
            mock.patch.object(main, "is_market_open", side_effect=lambda sym, _: (sym == "BTCUSD", "ok")),
            mock.patch.object(main, "get_candles", return_value=df),
            mock.patch.object(main, "update_open_signals"),
            mock.patch.object(main, "predict_future", side_effect=fake_predict),
            mock.patch.object(main, "create_signal_chart", return_value=None),
            mock.patch.object(main, "add_new_signal", return_value=(True, "BTC_1")) as add_spy,
            mock.patch.object(main, "send_telegram_message"),
            mock.patch.object(main, "send_telegram_photo"),
        ):
            main.run_single_cycle(manual_force=False, now_vn=datetime(2026, 11, 14, 14, 2, tzinfo=VN_TZ))

        self.assertEqual(captured["last_closed_time"], "2026-11-14T06:45:00+00:00")
        self.assertEqual(captured["live_price"], 70050.0)
        self.assertEqual(add_spy.call_args.kwargs["time_str"], "2026-11-14T06:45:00+00:00")

    def test_workflow_python_312_concurrency_and_rebase(self):
        wf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".github", "workflows", "signal_bot.yml")
        with open(wf_path, "r", encoding="utf-8") as f:
            yml = f.read()
        for snippet in (
            "python-version: '3.12'",
            "concurrency:",
            "cancel-in-progress: false",
            "git diff --cached --quiet",
            "git pull --rebase origin main",
            "HF_TOKEN:",
            "TELEGRAM_BOT_TOKEN:",
            "TELEGRAM_GROUP_ID:",
        ):
            self.assertIn(snippet, yml)

    def test_requirements_pins_pandas_v2_and_six(self):
        req_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements.txt")
        with open(req_path, "r", encoding="utf-8") as f:
            reqs = f.read()
        self.assertIn("pandas>=2.0.0,<3.0.0", reqs)
        self.assertIn("six>=1.16.0", reqs)


if __name__ == "__main__":
    unittest.main()
