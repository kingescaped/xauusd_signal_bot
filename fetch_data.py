import math
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf
import config
from market_time import VN_TZ, timeframe_to_timedelta, to_utc_datetime, to_vn_datetime


def _normalize_time_column_vn(df):
    raw_times = df["time"]
    if hasattr(raw_times, "dt") and getattr(raw_times.dt, "tz", None) is not None:
        df["time"] = raw_times.dt.tz_convert(VN_TZ)
        return df
    converted = [
        to_vn_datetime(raw_times.iloc[i], default_tz=timezone.utc) or raw_times.iloc[i]
        for i in range(len(raw_times))
    ]
    df["time"] = converted
    return df


def split_closed_and_live(df, timeframe="15m", now_utc=None):
    if df is None or len(df) == 0:
        return df, None, None, False

    now_utc = to_utc_datetime(now_utc) if now_utc is not None else datetime.now(timezone.utc)
    bar_delta = timeframe_to_timedelta(timeframe)
    last_row = df.iloc[-1]
    live_time_str = str(last_row["time"])
    raw_close = float(last_row["close"])
    live_price = None if math.isnan(raw_close) else raw_close

    end_idx = len(df)
    while end_idx > 0:
        bar_utc = to_utc_datetime(df.iloc[end_idx - 1]["time"], default_tz=timezone.utc)
        if bar_utc is not None and bar_utc + bar_delta > now_utc:
            end_idx -= 1
        else:
            break

    had_forming = end_idx < len(df)
    closed_df = df.iloc[:end_idx].copy() if had_forming else df.copy()
    if live_price is None and len(closed_df) > 0:
        for idx in range(len(closed_df) - 1, -1, -1):
            candidate = float(closed_df.iloc[idx]["close"])
            if not math.isnan(candidate):
                live_price = candidate
                break
    return closed_df, live_price, live_time_str, had_forming


def get_candles(ticker, timeframe="15m", period="5d", limit=config.CANDLE_LIMIT):
    try:
        df = yf.download(tickers=ticker, period=period, interval=timeframe, progress=False)
        if df.empty:
            print(f"[CẢNH BÁO] Không lấy được dữ liệu nến cho mã {ticker} ({timeframe})!")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
        })
        df = df[["open", "high", "low", "close", "volume"]].tail(limit).copy()
        df.reset_index(inplace=True)
        df.rename(columns={df.columns[0]: "time"}, inplace=True)
        return _normalize_time_column_vn(df)
    except Exception as e:
        print(f"[LỖI] Kéo nến {ticker} ({timeframe}) thất bại: {e}")
        return None


if __name__ == "__main__":
    for item in config.SYMBOLS:
        digits = int(item.get("digits", 2))
        print(f"Kiểm tra kéo nến {item['name']} ({item['ticker']}) khung {item['timeframe']}...")
        candles = get_candles(item["ticker"], timeframe=item["timeframe"], period=item["period"])
        if candles is not None:
            print(f"-> Thành công! Nến mới nhất: {candles['time'].iloc[-1]} | Giá Close: {candles['close'].iloc[-1]:.{digits}f}")
        else:
            print("-> Thất bại!")
