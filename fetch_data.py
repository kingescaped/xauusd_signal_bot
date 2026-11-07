import yfinance as yf
import pandas as pd
from datetime import datetime
import config

def get_candles(ticker, timeframe=config.TIMEFRAME, limit=config.CANDLE_LIMIT):
    """
    Kéo dữ liệu nến cho bất kỳ mã nào (Vàng 'GC=F', Bitcoin 'BTC-USD') từ Yahoo Finance.
    """
    try:
        # Với khung 15m, Yahoo Finance hỗ trợ tối đa period='60d' (dùng 5d là rất nhanh và đủ)
        period = "5d" if timeframe == "15m" else "14d"
        
        df = yf.download(tickers=ticker, period=period, interval=timeframe, progress=False)
        
        if df.empty:
            print(f"[CẢNH BÁO] Không lấy được dữ liệu nến cho mã {ticker}!")
            return None
        
        # Xử lý MultiIndex nếu có
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        df = df[['open', 'high', 'low', 'close', 'volume']].tail(limit).copy()
        df.reset_index(inplace=True)
        
        # Chuẩn hóa tên cột thời gian về 'time'
        time_col = df.columns[0]
        df.rename(columns={time_col: 'time'}, inplace=True)
        
        return df

    except Exception as e:
        print(f"[LỖI] Kéo nến {ticker} thất bại: {e}")
        return None

if __name__ == "__main__":
    for item in config.SYMBOLS:
        print(f"Kiểm tra kéo nến {item['name']} ({item['ticker']}) khung {config.TIMEFRAME}...")
        c = get_candles(item['ticker'])
        if c is not None:
            print(f"-> Thành công! Nến mới nhất: {c['time'].iloc[-1]} | Giá Close: {c['close'].iloc[-1]:.2f}")
        else:
            print("-> Thất bại!")
