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
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            try:
                import huggingface_hub
                huggingface_hub.login(token=hf_token, add_to_git_credential=False)
            except Exception:
                pass
        
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

def predict_future(df, symbol_name, timeframe="15m", pred_len=12):
    """
    Phân tích & dự báo xu hướng bằng KRONOS-BASE:
    - Bitcoin: Khung M15 (dự phóng 12 nến = 3 tiếng)
    - Vàng: Khung 1H (dự phóng 6 nến = 6 tiếng)
    """
    current_price = float(df['close'].iloc[-1])
    atr = calculate_volatility(df)

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

            # Ngưỡng phát tín hiệu
            threshold = 0.12 if timeframe == "1h" else 0.25

            if pct_change >= threshold:
                direction = "🟢 BUY (TĂNG)"
                confidence = round(min(88.0, 72.0 + abs(pct_change) * 15.0), 1)
                tp = pred_max_high
                sl = current_price - max(atr * 1.0, (tp - current_price) * 0.6)
            elif pct_change <= -threshold:
                direction = "🔴 SELL (GIẢM)"
                confidence = round(min(88.0, 72.0 + abs(pct_change) * 15.0), 1)
                tp = pred_min_low
                sl = current_price + max(atr * 1.0, (current_price - tp) * 0.6)
            else:
                direction = "⚪ SIDEWAY (ĐI NGANG)"
                confidence = 52.0
                tp = current_price
                sl = current_price

            print(f"[AI KRONOS THÀNH CÔNG] {symbol_name} ({timeframe}): {direction} | Dự phóng: {pct_change:+.2f}% | Độ tin cậy: {confidence}%")

            return {
                "current_price": round(current_price, 2),
                "direction": direction,
                "confidence": confidence,
                "tp": round(tp, 2),
                "sl": round(sl, 2),
                "atr": atr,
                "ai_predicted": True,
                "pct_change": round(pct_change, 2)
            }

        except Exception as e:
            print(f"[CẢNH BÁO AI] Lỗi trong quá trình suy luận Kronos: {e}")

    # Fallback
    ma_short = df['close'].tail(7).mean()
    ma_long = df['close'].tail(21).mean()
    momentum = current_price - df['close'].iloc[-4]
    
    if ma_short > ma_long and momentum > 0:
        direction = "🟢 BUY (TĂNG)"
        confidence = 72.0
        tp = current_price + (atr * 1.8)
        sl = current_price - (atr * 1.0)
    elif ma_short < ma_long and momentum < 0:
        direction = "🔴 SELL (GIẢM)"
        confidence = 72.0
        tp = current_price - (atr * 1.8)
        sl = current_price + (atr * 1.0)
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
        "pct_change": 0.0
    }
