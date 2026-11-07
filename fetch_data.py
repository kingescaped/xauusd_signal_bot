import yfinance as yf
import pandas as pd
from datetime import datetime
import config

def get_gold_candles(symbol=config.SYMBOL, timeframe=config.TIMEFRAME, limit=config.CANDLE_LIMIT):
    """
    Kéo dữ liệu nến Vàng thế giới từ Yahoo Finance.
    Khung '1h' sẽ lấy period='7d' để đủ số nến cần thiết.
    """
    try:
        # Ánh xạ interval sang period thích hợp của yfinance
        period_map = {
            "15m": "5d",
            "1h": "14d",
            "4h": "60d",
            "1d": "1y"
        }
        period = period_map.get(timeframe, "14d")
        
        # Tải dữ liệu từ Yahoo Finance
        df = yf.download(tickers=symbol, period=period, interval=timeframe, progress=False)
        
        if df.empty:
            print(f"[CẢNH BÁO] Không lấy được dữ liệu nến cho mã {symbol}!")
            return None
        
        # Xử lý đa chỉ mục (MultiIndex) nếu có trong các bản yfinance mới
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        # Chuẩn hóa tên cột
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Chỉ giữ lại các cột cần thiết và lấy số nến theo limit
        df = df[['open', 'high', 'low', 'close', 'volume']].tail(limit).copy()
        df.reset_index(inplace=True)
        
        # Đổi tên cột thời gian về 'time'
        time_col = df.columns[0]
        df.rename(columns={time_col: 'time'}, inplace=True)
        
        return df

    except Exception as e:
        print(f"[LỖI] Xảy ra lỗi khi kéo dữ liệu nến: {e}")
        return None

if __name__ == "__main__":
    print(f"Đang kiểm tra kéo nến Vàng ({config.SYMBOL}) khung {config.TIMEFRAME}...")
    candles = get_gold_candles()
    if candles is not None:
        print("Kéo nến thành công! Dữ liệu 5 cây nến mới nhất:")
        print(candles.tail())
    else:
        print("Kéo nến thất bại!")
