import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta
import config

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

_GLOBAL_PREDICTOR = None
_MODEL_LOADED = False


def _resolve_hf_token():
    hf_token = getattr(config, "HF_TOKEN", None) or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    hf_token = str(hf_token).strip() if hf_token else ""
    if not hf_token:
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        return None
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    try:
        import huggingface_hub
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)
    except Exception:
        pass
    return hf_token


def load_kronos_base():
    """Tải mô hình Kronos-base và Kronos-Tokenizer-base từ Hugging Face."""
    global _GLOBAL_PREDICTOR, _MODEL_LOADED
    if _GLOBAL_PREDICTOR is not None:
        return _GLOBAL_PREDICTOR

    try:
        print("\n[AI KRONOS] Đang khởi động mô hình Kronos-base...")
        from model import Kronos, KronosTokenizer, KronosPredictor

        tokenizer_name = getattr(config, "KRONOS_TOKENIZER_NAME", "NeoQuasar/Kronos-Tokenizer-base")
        model_name = getattr(config, "KRONOS_MODEL_NAME", "NeoQuasar/Kronos-base")
        hf_token = _resolve_hf_token()

        print(f"[AI KRONOS] Tải Tokenizer: {tokenizer_name}")
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name, token=hf_token)
        tokenizer.eval()

        print(f"[AI KRONOS] Tải Mô hình: {model_name}")
        model = Kronos.from_pretrained(model_name, token=hf_token)
        model.eval()

        _GLOBAL_PREDICTOR = KronosPredictor(model=model, tokenizer=tokenizer, device="cpu", max_context=512)
        _MODEL_LOADED = True
        print("[AI KRONOS] Đã nạp thành công Kronos-base vào bộ nhớ!\n")
        return _GLOBAL_PREDICTOR
    except Exception as e:
        print(f"[CẢNH BÁO AI] Không nạp được Kronos-base ({e}). Tạm dừng bắn tín hiệu tự động.")
        _MODEL_LOADED = False
        return None


def _get_symbol_meta(symbol_name, timeframe):
    default_thresh = getattr(config, "GOLD_THRESHOLD", 0.30) if (timeframe.lower() == "1h" or symbol_name == "XAUUSD") else getattr(config, "BTC_THRESHOLD", 0.55)
    for item in getattr(config, "SYMBOLS", []):
        if item.get("name") == symbol_name:
            return int(item.get("digits", 2)), float(item.get("threshold", default_thresh))
    return 2, float(default_thresh)


def calculate_volatility(df, window=14, digits=2):
    """Tính độ biến động ATR theo chuẩn Wilder's RMA."""
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / window, adjust=False).mean().iloc[-1]
    return round(float(atr), digits) if not np.isnan(atr) else 5.0


def calculate_ema(series, span):
    """Tính đường Exponential Moving Average (EMA)."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_adx(df, window=14):
    """Tính chỉ số ADX theo chuẩn Wilder's RMA."""
    try:
        if len(df) < window * 2:
            return 0.0
        high, low, close = df["high"], df["low"], df["close"]
        up_move, down_move = high.diff(), -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        prev_c = close.shift(1)
        tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / window, adjust=False).mean().replace(0, np.nan)
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / window, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / window, adjust=False).mean() / atr)

        denom = (plus_di + minus_di).replace(0, np.nan)
        dx = (100 * ((plus_di - minus_di).abs() / denom)).fillna(0.0)
        adx = dx.ewm(alpha=1.0 / window, adjust=False).mean().iloc[-1]
        return round(float(adx), 2) if not np.isnan(adx) else 0.0
    except Exception:
        return 0.0


def _clean_candle_df(df):
    cleaned = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if "volume" in cleaned.columns:
        cleaned["volume"] = cleaned["volume"].fillna(0.0)
    return cleaned


def _prepare_kronos_inputs(clean_df, timeframe, pred_len):
    lookback = min(len(clean_df), getattr(config, "CANDLE_LIMIT", 200))
    context_df = clean_df.tail(lookback).copy()
    x_timestamp = pd.to_datetime(context_df["time"])
    is_hourly = timeframe.lower() == "1h"
    step_delta = timedelta(hours=1) if is_hourly else timedelta(minutes=15)
    y_timestamp = pd.date_range(start=x_timestamp.iloc[-1] + step_delta, periods=pred_len, freq="1h" if is_hourly else "15min")

    x_df = context_df[["open", "high", "low", "close", "volume"]].copy()
    if "amount" not in x_df.columns:
        x_df["amount"] = x_df["volume"] * x_df[["open", "high", "low", "close"]].mean(axis=1)
    return x_df, x_timestamp, y_timestamp


def _run_kronos_predict(predictor, x_df, x_timestamp, y_timestamp, pred_len):
    kwargs = {
        "df": x_df, "x_timestamp": x_timestamp, "y_timestamp": y_timestamp, "pred_len": pred_len,
        "T": getattr(config, "KRONOS_TEMPERATURE", 0.6), "top_p": getattr(config, "KRONOS_TOP_P", 0.9),
        "sample_count": getattr(config, "KRONOS_SAMPLE_COUNT", 5), "verbose": False,
    }
    try:
        res = predictor.predict(**kwargs, return_samples=True)
    except TypeError:
        res = predictor.predict(**kwargs)
    return res if isinstance(res, tuple) and len(res) == 2 else (res, [res])


def _compute_mc_confidence(cand_direction, pct_change, ref_close, sample_dfs):
    if "SIDEWAY" in cand_direction or ref_close <= 0:
        return 50.0, 0.0, 0.0
    sample_pcts = [(float(s["close"].iloc[-1]) - ref_close) / ref_close * 100.0 for s in (sample_dfs or [])] or [pct_change]
    aligned = sum(1 for p in sample_pcts if (p > 0 if "BUY" in cand_direction else p < 0))
    agreement_ratio = aligned / len(sample_pcts)
    std_pct = float(np.std(sample_pcts))
    snr_score = min(1.0, (abs(pct_change) / (std_pct + 0.15)) / 2.5)
    conf = round(min(95.0, 45.0 + 35.0 * agreement_ratio + 15.0 * snr_score), 1)
    return conf, round(agreement_ratio * 100.0, 1), round(std_pct, 3)


def _apply_entry_slippage_and_targets(symbol_name, direction, conf, ref_close, entry_price, pred_high, pred_low, atr, sl_dist, min_tp_dist):
    if "SIDEWAY" in direction:
        return direction, conf, entry_price, entry_price, False
    max_slip = getattr(config, "MAX_ENTRY_SLIPPAGE_ATR", 0.35)
    drift = abs(entry_price - ref_close)
    if atr > 0 and drift > max_slip * atr:
        print(f"[BỘ LỌC TRƯỢT GIÁ] {symbol_name}: Giá hiện tại ({entry_price:.2f}) lệch {drift:.2f} > {max_slip}*ATR ({max_slip * atr:.2f}) so với nến đóng ({ref_close:.2f}) -> Bỏ kèo!")
        return "⚪ SIDEWAY (ĐI NGANG)", 51.0, entry_price, entry_price, True
    if "BUY" in direction:
        return direction, conf, entry_price + max(pred_high - entry_price, min_tp_dist), entry_price - sl_dist, False
    return direction, conf, entry_price - max(entry_price - pred_low, min_tp_dist), entry_price + sl_dist, False


def _resolve_direction_and_targets(symbol_name, ref_close, entry_price, pred_df, sample_dfs, threshold, atr, sl_dist, min_tp_dist, adx_val, ema_50, ema_200):
    pred_end = float(pred_df["close"].iloc[-1])
    pred_high, pred_low = float(pred_df["high"].max()), float(pred_df["low"].min())
    pct_change = (pred_end - ref_close) / ref_close * 100.0
    cand = "🟢 BUY (TĂNG)" if pct_change >= threshold else ("🔴 SELL (GIẢM)" if pct_change <= -threshold else "⚪ SIDEWAY (ĐI NGANG)")
    conf, agree_pct, std_pct = _compute_mc_confidence(cand, pct_change, ref_close, sample_dfs)

    if getattr(config, "USE_SIDEWAY_FILTER", True) and adx_val < getattr(config, "ADX_THRESHOLD", 20.0):
        print(f"[BỘ LỌC SIDEWAY] {symbol_name}: ADX = {adx_val:.1f} -> Thị trường đi ngang tích lũy, giữ im lặng.")
        return "⚪ SIDEWAY (ĐI NGANG)", 50.0, entry_price, entry_price, pct_change, pred_end, pred_high, pred_low, agree_pct, std_pct, False

    if getattr(config, "USE_TREND_FILTER", True):
        if cand == "🟢 BUY (TĂNG)" and (ref_close < ema_50 or ref_close < ema_200):
            print(f"[BỘ LỌC XU HƯỚNG EMA] {symbol_name}: Giá ({ref_close}) < EMA50/200 ({ema_50:.1f}/{ema_200:.1f}) -> Bỏ qua BUY để tránh bắt dao rơi")
            return "⚪ SIDEWAY (ĐI NGANG)", 52.0, entry_price, entry_price, pct_change, pred_end, pred_high, pred_low, agree_pct, std_pct, False
        if cand == "🔴 SELL (GIẢM)" and (ref_close > ema_50 or ref_close > ema_200):
            print(f"[BỘ LỌC XU HƯỚNG EMA] {symbol_name}: Giá ({ref_close}) > EMA50/200 ({ema_50:.1f}/{ema_200:.1f}) -> Bỏ qua SELL để tránh chặn đầu xe lửa")
            return "⚪ SIDEWAY (ĐI NGANG)", 52.0, entry_price, entry_price, pct_change, pred_end, pred_high, pred_low, agree_pct, std_pct, False

    direction, conf, tp, sl, slip_blocked = _apply_entry_slippage_and_targets(
        symbol_name, cand, conf, ref_close, entry_price, pred_high, pred_low, atr, sl_dist, min_tp_dist,
    )
    return direction, conf, tp, sl, pct_change, pred_end, pred_high, pred_low, agree_pct, std_pct, slip_blocked


def _no_ai_result(entry_price, ref_close, atr, adx_val, ema_50, ema_200, digits=2):
    return {
        "current_price": round(entry_price, digits), "ref_close": round(ref_close, digits),
        "direction": "⚪ SIDEWAY (ĐI NGANG)", "confidence": 0.0,
        "tp": round(entry_price, digits), "sl": round(entry_price, digits),
        "atr": atr, "ai_predicted": False, "slippage_blocked": False, "pct_change": 0.0,
        "pred_end_price": round(ref_close, digits), "pred_high": round(ref_close, digits), "pred_low": round(ref_close, digits),
        "adx": adx_val, "ema_50": round(ema_50, digits), "ema_200": round(ema_200, digits),
        "sample_agreement": 0.0, "sample_std_pct": 0.0, "strategy_version": getattr(config, "STRATEGY_VERSION", "v3.2-p1-lite"),
    }


def predict_future(df, symbol_name, timeframe="15m", pred_len=12, live_price=None):
    """Phân tích & dự báo xu hướng bằng KRONOS-BASE kết hợp EMA, Wilder ATR/ADX, MC & Trượt giá."""
    digits, threshold = _get_symbol_meta(symbol_name, timeframe)
    clean_df = _clean_candle_df(df)
    if clean_df.empty:
        return _no_ai_result(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, digits)

    ref_close = float(clean_df["close"].iloc[-1])
    entry_price = float(live_price) if (live_price is not None and not np.isnan(live_price)) else ref_close
    atr = calculate_volatility(clean_df, digits=digits)
    sl_dist = atr * getattr(config, "ATR_SL_MULTIPLIER", 1.5)
    min_tp_dist = sl_dist * getattr(config, "MIN_RR_RATIO", 1.2)
    adx_val = calculate_adx(clean_df)
    ema_50 = float(calculate_ema(clean_df["close"], 50).iloc[-1])
    ema_200 = float(calculate_ema(clean_df["close"], min(len(clean_df), 200)).iloc[-1])

    predictor = load_kronos_base()
    if predictor is not None and len(clean_df) >= 25:
        try:
            print(f"[AI KRONOS] Đang chạy suy luận K-line cho {symbol_name} ({timeframe})...")
            x_df, x_timestamp, y_timestamp = _prepare_kronos_inputs(clean_df, timeframe, pred_len)
            pred_df, sample_dfs = _run_kronos_predict(predictor, x_df, x_timestamp, y_timestamp, pred_len)
            direction, conf, tp, sl, pct_change, p_end, p_high, p_low, agree, std_pct, slip_blocked = _resolve_direction_and_targets(
                symbol_name, ref_close, entry_price, pred_df, sample_dfs, threshold, atr, sl_dist, min_tp_dist, adx_val, ema_50, ema_200,
            )
            print(f"[AI KRONOS PHÂN TÍCH] {symbol_name} ({timeframe}): {direction} | Dự phóng: {pct_change:+.2f}% | Đồng thuận: {agree}% | ADX: {adx_val:.1f} | Tin cậy: {conf}%")
            return {
                "current_price": round(entry_price, digits), "ref_close": round(ref_close, digits),
                "direction": direction, "confidence": conf, "tp": round(tp, digits), "sl": round(sl, digits),
                "atr": atr, "ai_predicted": True, "slippage_blocked": slip_blocked,
                "pct_change": round(pct_change, 2), "pred_end_price": round(p_end, digits),
                "pred_high": round(p_high, digits), "pred_low": round(p_low, digits),
                "adx": adx_val, "ema_50": round(ema_50, digits), "ema_200": round(ema_200, digits),
                "sample_agreement": agree, "sample_std_pct": std_pct,
                "strategy_version": getattr(config, "STRATEGY_VERSION", "v3.2-p1-lite"),
            }
        except Exception as e:
            print(f"[CẢNH BÁO AI] Lỗi trong quá trình suy luận Kronos: {e}")

    return _no_ai_result(entry_price, ref_close, atr, adx_val, ema_50, ema_200, digits)
