import pandas as pd
import numpy as np
import config

def calculate_volatility(df, window=14):
    """Tính độ biến động trung bình (ATR) để gợi ý SL/TP theo giá Vàng."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean().iloc[-1]
    return round(float(atr), 2) if not np.isnan(atr) else 5.0

def predict_future(df):
    """
    Module phân tích & dự báo xu hướng nến Vàng:
    - Đầu vào: DataFrame nến OHLCV (open, high, low, close, volume).
    - Đầu ra: Hướng dự báo, độ tin cậy (%), Giá kỳ vọng (TP), Cắt lỗ (SL).
    """
    current_price = float(df['close'].iloc[-1])
    atr = calculate_volatility(df)
    
    # -------------------------------------------------------------------------
    # GHI CHÚ TÍCH HỢP KRONOS FOUNDATION MODEL:
    # Khi bạn clone repo shiyu-coder/Kronos và tải checkpoint weights về,
    # mã nguồn tại đây sẽ gọi:
    #   tokens = kronos_tokenizer.encode(df[['open','high','low','close','volume']])
    #   prediction = kronos_model.generate(tokens, num_samples=100)
    # -------------------------------------------------------------------------
    
    # Logic phân tích động lượng & nến gần nhất (làm cơ sở baseline)
    ma_short = df['close'].tail(9).mean()
    ma_long = df['close'].tail(21).mean()
    momentum = current_price - df['close'].iloc[-5]
    
    if ma_short > ma_long and momentum > 0:
        direction = "🟢 BUY (TĂNG)"
        confidence = round(70.0 + min(abs(momentum) * 1.5, 22.0), 1)
        tp = current_price + (atr * 1.8)
        sl = current_price - (atr * 1.0)
    elif ma_short < ma_long and momentum < 0:
        direction = "🔴 SELL (GIẢM)"
        confidence = round(70.0 + min(abs(momentum) * 1.5, 22.0), 1)
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
        "atr": atr
    }
