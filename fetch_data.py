import yfinance as yf
import pandas as pd
import config

def get_candles(ticker, timeframe="15m", period="5d", limit=config.CANDLE_LIMIT):
    """
    Kéo dữ liệu nến cho từng mã với khung thời gian (timeframe) riêng biệt.
    - BTC: 15m (period 5d)
    - Vàng: 1h (period 14d)
    """
    try:
        df = yf.download(tickers=ticker, period=period, interval=timeframe, progress=False)
        
        if df.empty:
            print(f"[CẢNH BÁO] Không lấy được dữ liệu nến cho mã {ticker} ({timeframe})!")
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
        print(f"[LỖI] Kéo nến {ticker} ({timeframe}) thất bại: {e}")
        return None

if __name__ == "__main__":
    for item in config.SYMBOLS:
        print(f"Kiểm tra kéo nến {item['name']} ({item['ticker']}) khung {item['timeframe']}...")
        c = get_candles(item['ticker'], timeframe=item['timeframe'], period=item['period'])
        if c is not None:
            print(f"-> Thành công! Nến mới nhất: {c['time'].iloc[-1]} | Giá Close: {c['close'].iloc[-1]:.2f}")
        else:
            print("-> Thất bại!")
