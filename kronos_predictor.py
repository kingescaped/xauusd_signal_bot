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
        hf_token = getattr(config, "HF_TOKEN", None) or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

        if hf_token:
            hf_token = str(hf_token).strip()
            if not hf_token:
                hf_token = None

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            try:
                import huggingface_hub
                huggingface_hub.login(token=hf_token, add_to_git_credential=False)
            except Exception:
                pass
        else:
            hf_token = None
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        
        print(f"[AI KRONOS] Tải Tokenizer: {tokenizer_name}")
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name, token=hf_token)
        
        print(f"[AI KRONOS] Tải Mô hình: {model_name}")
        model = Kronos.from_pretrained(model_name, token=hf_token)
        
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
        print(f"[CẢNH BÁO AI] Không nạp được Kronos-base ({e}). Dùng chế độ phân tích kỹ thuật dự phòng.")
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

def predict_future(df, symbol_name, timeframe="15m", pred_len=12):
    """
    Phân tích & dự báo xu hướng bằng KRONOS-BASE kết hợp Bộ Lọc EMA & Sideway:
    - Bitcoin: Khung M15 (dự phóng 12 nến = 3 tiếng)
    - Vàng: Khung 1H (dự phóng 6 nến = 6 tiếng)
    """
    current_price = float(df['close'].iloc[-1])
    atr = calculate_volatility(df)
    
    # Lấy các cấu hình quản lý rủi ro và ngưỡng kích hoạt
    atr_mult = getattr(config, 'ATR_SL_MULTIPLIER', 1.5)
    min_rr = getattr(config, 'MIN_RR_RATIO', 1.2)
    sl_dist = atr * atr_mult
    min_tp_dist = sl_dist * min_rr
    
    btc_thresh = getattr(config, 'BTC_THRESHOLD', 0.55)
    gold_thresh = getattr(config, 'GOLD_THRESHOLD', 0.30)
    threshold = gold_thresh if (timeframe.lower() == "1h" or symbol_name == "XAUUSD") else btc_thresh

    # Tính các chỉ báo kỹ thuật hỗ trợ
    adx_val = calculate_adx(df)
    ema_50 = float(calculate_ema(df['close'], 50).iloc[-1])
    ema_200 = float(calculate_ema(df['close'], min(len(df), 200)).iloc[-1])

    predictor = load_kronos_base()

    if predictor is not None:
        try:
            print(f"[AI KRONOS] Đang chạy suy luận K-line cho {symbol_name} ({timeframe})...")
            lookback = min(len(df), 200)
            context_df = df.tail(lookback).copy()
            
            x_timestamp = pd.to_datetime(context_df['time'])
            last_time = x_timestamp.iloc[-1]
            
            # Tính bước nhảy thời gian tương lai theo timeframe
            if timeframe.lower() == "1h":
                step_delta = timedelta(hours=1)
                freq_str = '1h'
            else:
                step_delta = timedelta(minutes=15)
                freq_str = '15min'

            y_timestamp = pd.date_range(start=last_time + step_delta, periods=pred_len, freq=freq_str)

            x_df = context_df[['open', 'high', 'low', 'close', 'volume']].copy()
            if 'amount' not in x_df.columns:
                x_df['amount'] = x_df['volume'] * x_df[['open', 'high', 'low', 'close']].mean(axis=1)

            # Dự báo chuỗi K-line bằng Kronos-base
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=0.8,
                top_p=0.9,
                sample_count=3,
                verbose=False
            )

            pred_end_price = float(pred_df['close'].iloc[-1])
            pred_max_high = float(pred_df['high'].max())
            pred_min_low = float(pred_df['low'].min())
            
            pct_change = (pred_end_price - current_price) / current_price * 100

            # 1. Kiểm tra hướng dự báo từ AI Kronos
            cand_direction = "⚪ SIDEWAY (ĐI NGANG)"
            if pct_change >= threshold:
                cand_direction = "🟢 BUY (TĂNG)"
            elif pct_change <= -threshold:
                cand_direction = "🔴 SELL (GIẢM)"

            # 2. Áp dụng Bộ Lọc Sideway (ADX)
            use_sideway_filter = getattr(config, 'USE_SIDEWAY_FILTER', True)
            adx_threshold = getattr(config, 'ADX_THRESHOLD', 20.0)
            is_sideway = use_sideway_filter and (adx_val < adx_threshold)

            # 3. Áp dụng Bộ Lọc Xu Hướng EMA (EMA 50 & EMA 200)
            use_trend_filter = getattr(config, 'USE_TREND_FILTER', True)
            trend_conflict = False
            trend_reason = ""
            
            if use_trend_filter and cand_direction == "🟢 BUY (TĂNG)":
                if current_price < ema_50 or current_price < ema_200:
                    trend_conflict = True
                    trend_reason = f"Giá ({current_price}) < EMA50/200 ({ema_50:.1f}/{ema_200:.1f}) -> Bỏ qua BUY để tránh bắt dao rơi"
            elif use_trend_filter and cand_direction == "🔴 SELL (GIẢM)":
                if current_price > ema_50 or current_price > ema_200:
                    trend_conflict = True
                    trend_reason = f"Giá ({current_price}) > EMA50/200 ({ema_50:.1f}/{ema_200:.1f}) -> Bỏ qua SELL để tránh chặn đầu xe lửa"

            # 4. Quyết định hướng lệnh cuối cùng và tính TP/SL
            if is_sideway:
                direction = "⚪ SIDEWAY (ĐI NGANG)"
                confidence = 50.0
                tp = current_price
                sl = current_price
                print(f"[BỘ LỌC SIDEWAY] {symbol_name}: ADX = {adx_val:.1f} < {adx_threshold} -> Thị trường đi ngang tích lũy, giữ im lặng.")
            elif trend_conflict:
                direction = "⚪ SIDEWAY (ĐI NGANG)"
                confidence = 52.0
                tp = current_price
                sl = current_price
                print(f"[BỘ LỌC XU HƯỚNG EMA] {symbol_name}: {trend_reason}")
            elif cand_direction == "🟢 BUY (TĂNG)":
                direction = "🟢 BUY (TĂNG)"
                excess_ratio = (pct_change - threshold) / threshold
                confidence = round(min(88.0, 71.0 + excess_ratio * 12.0), 1)
                ai_tp_dist = pred_max_high - current_price
                tp = current_price + max(ai_tp_dist, min_tp_dist)
                sl = current_price - sl_dist
            elif cand_direction == "🔴 SELL (GIẢM)":
                direction = "🔴 SELL (GIẢM)"
                excess_ratio = (abs(pct_change) - threshold) / threshold
                confidence = round(min(88.0, 71.0 + excess_ratio * 12.0), 1)
                ai_tp_dist = current_price - pred_min_low
                tp = current_price - max(ai_tp_dist, min_tp_dist)
                sl = current_price + sl_dist
            else:
                direction = "⚪ SIDEWAY (ĐI NGANG)"
                confidence = 50.0
                tp = current_price
                sl = current_price

            print(f"[AI KRONOS PHÂN TÍCH] {symbol_name} ({timeframe}): {direction} | Dự phóng: {pct_change:+.2f}% | ADX: {adx_val:.1f} | EMA50: {ema_50:.1f} | Độ tin cậy: {confidence}%")

            return {
                "current_price": round(current_price, 2),
                "direction": direction,
                "confidence": confidence,
                "tp": round(tp, 2),
                "sl": round(sl, 2),
                "atr": atr,
                "ai_predicted": True,
                "pct_change": round(pct_change, 2),
                "adx": adx_val,
                "ema_50": round(ema_50, 2),
                "ema_200": round(ema_200, 2)
            }

        except Exception as e:
            print(f"[CẢNH BÁO AI] Lỗi trong quá trình suy luận Kronos: {e}")

    # Fallback khi mô hình AI lỗi hoặc không tải được
    ma_short = df['close'].tail(7).mean()
    ma_long = df['close'].tail(21).mean()
    momentum = current_price - df['close'].iloc[-4]
    
    # Kiểm tra bộ lọc xu hướng cho Fallback
    if ma_short > ma_long and momentum > 0 and current_price >= ema_50 and adx_val >= 20.0:
        direction = "🟢 BUY (TĂNG)"
        confidence = 75.0
        tp = current_price + min_tp_dist
        sl = current_price - sl_dist
    elif ma_short < ma_long and momentum < 0 and current_price <= ema_50 and adx_val >= 20.0:
        direction = "🔴 SELL (GIẢM)"
        confidence = 75.0
        tp = current_price - min_tp_dist
        sl = current_price + sl_dist
    else:
        direction = "⚪ SIDEWAY (ĐI NGANG)"
        confidence = 50.0
        tp = current_price
        sl = current_price

    return {
        "current_price": round(current_price, 2),
        "direction": direction,
        "confidence": confidence,
        "tp": round(tp, 2),
        "sl": round(sl, 2),
        "atr": atr,
        "ai_predicted": False,
        "pct_change": 0.0,
        "adx": adx_val,
        "ema_50": round(ema_50, 2),
        "ema_200": round(ema_200, 2)
    }
