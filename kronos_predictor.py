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

        _GLOBAL_PREDICTOR = KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            max_context=512
        )
        _MODEL_LOADED = True
        print("[AI KRONOS] Đã nạp thành công Kronos-base vào bộ nhớ!\n")
        return _GLOBAL_PREDICTOR

    except Exception as e:
        print(f"[CẢNH BÁO AI] Không nạp được Kronos-base ({e}). Tạm dừng bắn tín hiệu tự động.")
        _MODEL_LOADED = False
        return None

def calculate_volatility(df, window=14):
    """Tính độ biến động ATR."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window).mean().iloc[-1]
    return round(float(atr), 2) if not np.isnan(atr) else 5.0

def calculate_ema(series, span):
    """Tính đường Exponential Moving Average (EMA)."""
    return series.ewm(span=span, adjust=False).mean()

def calculate_adx(df, window=14):
    """Tính chỉ số ADX để xác định thị trường có xu hướng hay đang Sideway tích lũy."""
    try:
        if len(df) < window * 2:
            return 25.0
        high = df['high']
        low = df['low']
        close = df['close']

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=window).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=window).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=window).mean() / atr)

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.rolling(window=window).mean().iloc[-1]
        return round(float(adx), 2) if not np.isnan(adx) else 25.0
    except Exception:
        return 25.0

def _clean_candle_df(df):
    cleaned = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if "volume" in cleaned.columns:
        cleaned["volume"] = cleaned["volume"].fillna(0.0)
    return cleaned


def _prepare_kronos_inputs(clean_df, timeframe, pred_len):
    lookback = min(len(clean_df), 200)
    context_df = clean_df.tail(lookback).copy()
    x_timestamp = pd.to_datetime(context_df["time"])
    last_time = x_timestamp.iloc[-1]
    is_hourly = timeframe.lower() == "1h"
    step_delta = timedelta(hours=1) if is_hourly else timedelta(minutes=15)
    freq_str = "1h" if is_hourly else "15min"
    y_timestamp = pd.date_range(start=last_time + step_delta, periods=pred_len, freq=freq_str)

    x_df = context_df[["open", "high", "low", "close", "volume"]].copy()
    if "amount" not in x_df.columns:
        x_df["amount"] = x_df["volume"] * x_df[["open", "high", "low", "close"]].mean(axis=1)
    return x_df, x_timestamp, y_timestamp


def _apply_entry_slippage_and_targets(symbol_name, direction, conf, ref_close, entry_price, pred_max_high, pred_min_low, atr, sl_dist, min_tp_dist):
    if "SIDEWAY" in direction:
        return direction, conf, entry_price, entry_price, False

    max_slip = getattr(config, "MAX_ENTRY_SLIPPAGE_ATR", 0.35)
    drift = abs(entry_price - ref_close)
    if atr > 0 and drift > max_slip * atr:
        print(f"[BỘ LỌC TRƯỢT GIÁ] {symbol_name}: Giá hiện tại ({entry_price:.2f}) lệch {drift:.2f} > {max_slip}*ATR ({max_slip * atr:.2f}) so với nến đóng ({ref_close:.2f}) -> Bỏ kèo!")
        return "⚪ SIDEWAY (ĐI NGANG)", 51.0, entry_price, entry_price, True

    if "BUY" in direction:
        tp = entry_price + max(pred_max_high - entry_price, min_tp_dist)
        return direction, conf, tp, entry_price - sl_dist, False
    tp = entry_price - max(entry_price - pred_min_low, min_tp_dist)
    return direction, conf, tp, entry_price + sl_dist, False


def _resolve_direction_and_targets(symbol_name, ref_close, entry_price, pred_df, threshold, atr, sl_dist, min_tp_dist, adx_val, ema_50, ema_200):
    pred_end = float(pred_df["close"].iloc[-1])
    pred_max_high = float(pred_df["high"].max())
    pred_min_low = float(pred_df["low"].min())
    pct_change = (pred_end - ref_close) / ref_close * 100

    cand = "🟢 BUY (TĂNG)" if pct_change >= threshold else ("🔴 SELL (GIẢM)" if pct_change <= -threshold else "⚪ SIDEWAY (ĐI NGANG)")
    use_sw = getattr(config, "USE_SIDEWAY_FILTER", True)
    adx_thresh = getattr(config, "ADX_THRESHOLD", 20.0)
    if use_sw and adx_val < adx_thresh:
        print(f"[BỘ LỌC SIDEWAY] {symbol_name}: ADX = {adx_val:.1f} < {adx_thresh} -> Thị trường đi ngang tích lũy, giữ im lặng.")
        return "⚪ SIDEWAY (ĐI NGANG)", 50.0, entry_price, entry_price, pct_change, False

    if getattr(config, "USE_TREND_FILTER", True):
        if cand == "🟢 BUY (TĂNG)" and (ref_close < ema_50 or ref_close < ema_200):
            print(f"[BỘ LỌC XU HƯỚNG EMA] {symbol_name}: Giá ({ref_close}) < EMA50/200 ({ema_50:.1f}/{ema_200:.1f}) -> Bỏ qua BUY để tránh bắt dao rơi")
            return "⚪ SIDEWAY (ĐI NGANG)", 52.0, entry_price, entry_price, pct_change, False
        if cand == "🔴 SELL (GIẢM)" and (ref_close > ema_50 or ref_close > ema_200):
            print(f"[BỘ LỌC XU HƯỚNG EMA] {symbol_name}: Giá ({ref_close}) > EMA50/200 ({ema_50:.1f}/{ema_200:.1f}) -> Bỏ qua SELL để tránh chặn đầu xe lửa")
            return "⚪ SIDEWAY (ĐI NGANG)", 52.0, entry_price, entry_price, pct_change, False

    conf = 50.0
    if cand in ("🟢 BUY (TĂNG)", "🔴 SELL (GIẢM)"):
        excess = (abs(pct_change) - threshold) / threshold
        conf = round(min(88.0, 71.0 + excess * 12.0), 1)
    direction, conf, tp, sl, slip_blocked = _apply_entry_slippage_and_targets(
        symbol_name, cand, conf, ref_close, entry_price, pred_max_high, pred_min_low, atr, sl_dist, min_tp_dist,
    )
    return direction, conf, tp, sl, pct_change, slip_blocked


def _no_ai_result(entry_price, ref_close, atr, adx_val, ema_50, ema_200):
    return {
        "current_price": round(entry_price, 2),
        "ref_close": round(ref_close, 2),
        "direction": "⚪ SIDEWAY (ĐI NGANG)",
        "confidence": 0.0,
        "tp": round(entry_price, 2),
        "sl": round(entry_price, 2),
        "atr": atr,
        "ai_predicted": False,
        "slippage_blocked": False,
        "pct_change": 0.0,
        "adx": adx_val,
        "ema_50": round(ema_50, 2),
        "ema_200": round(ema_200, 2),
    }


def predict_future(df, symbol_name, timeframe="15m", pred_len=12, live_price=None):
    """Phân tích & dự báo xu hướng bằng KRONOS-BASE kết hợp Bộ Lọc EMA, Sideway & Trượt giá."""
    clean_df = _clean_candle_df(df)
    if clean_df.empty:
        return _no_ai_result(0.0, 0.0, 5.0, 25.0, 0.0, 0.0)

    ref_close = float(clean_df["close"].iloc[-1])
    entry_price = float(live_price) if (live_price is not None and not np.isnan(live_price)) else ref_close
    atr = calculate_volatility(clean_df)
    sl_dist = atr * getattr(config, "ATR_SL_MULTIPLIER", 1.5)
    min_tp_dist = sl_dist * getattr(config, "MIN_RR_RATIO", 1.2)
    threshold = getattr(config, "GOLD_THRESHOLD", 0.30) if (timeframe.lower() == "1h" or symbol_name == "XAUUSD") else getattr(config, "BTC_THRESHOLD", 0.55)
    adx_val = calculate_adx(clean_df)
    ema_50 = float(calculate_ema(clean_df["close"], 50).iloc[-1])
    ema_200 = float(calculate_ema(clean_df["close"], min(len(clean_df), 200)).iloc[-1])

    predictor = load_kronos_base()
    if predictor is not None and len(clean_df) >= 25:
        try:
            print(f"[AI KRONOS] Đang chạy suy luận K-line cho {symbol_name} ({timeframe})...")
            x_df, x_timestamp, y_timestamp = _prepare_kronos_inputs(clean_df, timeframe, pred_len)
            pred_df = predictor.predict(
                df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                pred_len=pred_len, T=0.8, top_p=0.9, sample_count=3, verbose=False,
            )
            direction, conf, tp, sl, pct_change, slip_blocked = _resolve_direction_and_targets(
                symbol_name, ref_close, entry_price, pred_df, threshold, atr, sl_dist, min_tp_dist, adx_val, ema_50, ema_200,
            )
            print(f"[AI KRONOS PHÂN TÍCH] {symbol_name} ({timeframe}): {direction} | Dự phóng: {pct_change:+.2f}% | ADX: {adx_val:.1f} | EMA50: {ema_50:.1f} | Độ tin cậy: {conf}%")
            return {
                "current_price": round(entry_price, 2), "ref_close": round(ref_close, 2),
                "direction": direction, "confidence": conf, "tp": round(tp, 2), "sl": round(sl, 2),
                "atr": atr, "ai_predicted": True, "slippage_blocked": slip_blocked,
                "pct_change": round(pct_change, 2), "adx": adx_val,
                "ema_50": round(ema_50, 2), "ema_200": round(ema_200, 2),
            }
        except Exception as e:
            print(f"[CẢNH BÁO AI] Lỗi trong quá trình suy luận Kronos: {e}")

    return _no_ai_result(entry_price, ref_close, atr, adx_val, ema_50, ema_200)
