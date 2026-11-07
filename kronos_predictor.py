import pandas as pd
import numpy as np
import config

def calculate_volatility(df, window=14):
    """Tính độ biến động trung bình (ATR) thích ứng cho cả Vàng và Bitcoin."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean().iloc[-1]
    return round(float(atr), 2) if not np.isnan(atr) else 5.0

def predict_future(df, symbol_name):
    """
    Phân tích & dự báo xu hướng nến M15 cho từng sản phẩm:
    - Vàng (XAUUSD): biến động vài giá ($/oz).
    - Bitcoin (BTCUSD): biến động vài trăm đến nghìn USD.
    """
    current_price = float(df['close'].iloc[-1])
    atr = calculate_volatility(df)
    
    # -------------------------------------------------------------------------
    # GHI CHÚ TÍCH HỢP KRONOS FOUNDATION MODEL:
    # Sau khi tải trọng số Kronos (shiyu-coder/Kronos),
    # đầu vào sẽ là df[['open', 'high', 'low', 'close', 'volume']]
    # để mô hình autoregressive sinh nến tương lai.
    # -------------------------------------------------------------------------

    # Phân tích kỹ thuật đa khung & động lượng làm baseline
    ma_short = df['close'].tail(7).mean()
    ma_long = df['close'].tail(21).mean()
    momentum = current_price - df['close'].iloc[-4]
    
    # Tỷ lệ R:R chuẩn (TP: 1.5 - 2.0 ATR, SL: 1.0 ATR)
    tp_multiplier = 1.8
    sl_multiplier = 1.0

    if ma_short > ma_long and momentum > 0:
        direction = "🟢 BUY (TĂNG)"
        confidence = round(70.0 + min(abs(momentum) / atr * 5.0, 24.0), 1)
        tp = current_price + (atr * tp_multiplier)
        sl = current_price - (atr * sl_multiplier)
    elif ma_short < ma_long and momentum < 0:
        direction = "🔴 SELL (GIẢM)"
        confidence = round(70.0 + min(abs(momentum) / atr * 5.0, 24.0), 1)
        tp = current_price - (atr * tp_multiplier)
        sl = current_price + (atr * sl_multiplier)
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
