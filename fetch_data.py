import math
from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd
import config


def _timeframe_to_timedelta(timeframe):
    tf = str(timeframe or "15m").strip().lower()
    if tf.endswith("m") and tf[:-1].isdigit():
        return timedelta(minutes=int(tf[:-1]))
    if tf.endswith("h") and tf[:-1].isdigit():
        return timedelta(hours=int(tf[:-1]))
    if tf.endswith("d") and tf[:-1].isdigit():
        return timedelta(days=int(tf[:-1]))
    return timedelta(minutes=15)


def _parse_bar_utc(val):
    if val is None:
        return None
    if hasattr(val, "to_pydatetime"):
        val = val.to_pydatetime()
    if isinstance(val, datetime):
        return (val.replace(tzinfo=timezone.utc) if val.tzinfo is None else val).astimezone(timezone.utc)
    try:
        dt = datetime.fromisoformat(str(val).strip())
        return (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt).astimezone(timezone.utc)
    except ValueError:
        return None


def split_closed_and_live(df, timeframe="15m", now_utc=None):
    """Separate completed candles (for AI/indicators) from current live price."""
    if df is None or len(df) == 0:
        return df, None, None, False

    now_utc = now_utc or datetime.now(timezone.utc)
    bar_delta = _timeframe_to_timedelta(timeframe)
    last_row = df.iloc[-1]
    live_time_str = str(last_row["time"])
    raw_close = float(last_row["close"])
    live_price = None if math.isnan(raw_close) else raw_close

    end_idx = len(df)
    while end_idx > 0:
        bar_utc = _parse_bar_utc(df.iloc[end_idx - 1]["time"])
        if bar_utc is not None and bar_utc + bar_delta > now_utc:
            end_idx -= 1
        else:
            break

    had_forming = end_idx < len(df)
    closed_df = df.iloc[:end_idx].copy() if had_forming else df.copy()
    if live_price is None and len(closed_df) > 0:
        fallback_close = float(closed_df.iloc[-1]["close"])
        live_price = None if math.isnan(fallback_close) else fallback_close
    return closed_df, live_price, live_time_str, had_forming


def get_candles(ticker, timeframe="15m", period="5d", limit=config.CANDLE_LIMIT):
    """Kéo dữ liệu nến cho từng mã với khung thời gian (timeframe) riêng biệt."""
    try:
        df = yf.download(tickers=ticker, period=period, interval=timeframe, progress=False)
        if df.empty:
            print(f"[CẢNH BÁO] Không lấy được dữ liệu nến cho mã {ticker} ({timeframe})!")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
        })
        df = df[["open", "high", "low", "close", "volume"]].tail(limit).copy()
        df.reset_index(inplace=True)
        df.rename(columns={df.columns[0]: "time"}, inplace=True)
        return df

    except Exception as e:
        print(f"[LỖI] Kéo nến {ticker} ({timeframe}) thất bại: {e}")
        return None


if __name__ == "__main__":
    for item in config.SYMBOLS:
        print(f"Kiểm tra kéo nến {item['name']} ({item['ticker']}) khung {item['timeframe']}...")
        c = get_candles(item["ticker"], timeframe=item["timeframe"], period=item["period"])
        if c is not None:
            print(f"-> Thành công! Nến mới nhất: {c['time'].iloc[-1]} | Giá Close: {c['close'].iloc[-1]:.2f}")
        else:
            print("-> Thất bại!")
