from datetime import datetime, timedelta, timezone
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fetch_data import split_closed_and_live
from model.kronos import calc_time_stamps

BINANCE_ENDPOINTS = (
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
)
PRICE_COLS = ("open", "high", "low", "close")
ALL_FEATURE_COLS = ("open", "high", "low", "close", "volume", "amount")
DEFAULT_FALLBACK_VAL_RATIO = 0.10


def _request_binance_batch(symbol, interval, start_ms, end_ms, limit=1000):
    params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": limit}
    last_err = None
    for url in BINANCE_ENDPOINTS:
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            last_err = RuntimeError(f"HTTP {resp.status_code} from {url}")
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Binance public klines fetch failed: {last_err}")


def _finalize_ohlcv_dataframe(df):
    cleaned = df.dropna(subset=list(PRICE_COLS)).copy()
    if "volume" not in cleaned.columns:
        cleaned["volume"] = 0.0
    cleaned["volume"] = cleaned["volume"].fillna(0.0).astype(float)
    for col in PRICE_COLS:
        cleaned[col] = cleaned[col].astype(float)
    cleaned["time"] = pd.to_datetime(cleaned["time"], utc=True)
    cleaned = cleaned.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    cleaned["amount"] = cleaned["volume"] * cleaned[list(PRICE_COLS)].mean(axis=1)
    return cleaned[["time", *ALL_FEATURE_COLS]]


def fetch_binance_btc_m15(days=365, symbol="BTCUSDT"):
    now_utc = datetime.now(timezone.utc)
    end_ms = int(now_utc.timestamp() * 1000)
    cursor_ms = int((now_utc - timedelta(days=int(days))).timestamp() * 1000)
    step_ms = 15 * 60 * 1000
    rows = []

    while cursor_ms < end_ms:
        batch = _request_binance_batch(symbol, "15m", cursor_ms, end_ms, limit=1000)
        if not batch:
            break
        for kline in batch:
            close_ms = int(kline[6])
            if close_ms < end_ms:
                rows.append({
                    "time": datetime.fromtimestamp(int(kline[0]) / 1000.0, tz=timezone.utc),
                    "open": float(kline[1]), "high": float(kline[2]),
                    "low": float(kline[3]), "close": float(kline[4]), "volume": float(kline[5]),
                })
        next_ms = int(batch[-1][0]) + step_ms
        if next_ms <= cursor_ms:
            break
        cursor_ms = next_ms
        time.sleep(0.08)

    if not rows:
        raise ValueError("No closed BTC M15 candles returned from Binance Public API")
    return _finalize_ohlcv_dataframe(pd.DataFrame(rows))


def load_exness_csv_h1(csv_path):
    raw = pd.read_csv(csv_path, sep=None, engine="python")
    norm_cols = {c: c.strip().strip("<>").lower() for c in raw.columns}
    raw = raw.rename(columns=norm_cols)
    if "date" in raw.columns and "time" in raw.columns:
        raw["time"] = raw["date"].astype(str) + " " + raw["time"].astype(str)
    vol_col = "tickvol" if "tickvol" in raw.columns else ("vol" if "vol" in raw.columns else "volume")
    if vol_col in raw.columns and vol_col != "volume":
        raw["volume"] = raw[vol_col]
    return _finalize_ohlcv_dataframe(raw)


def fetch_gold_h1_candles(period="730d", ticker="GC=F", exness_csv_path=""):
    if exness_csv_path:
        return load_exness_csv_h1(exness_csv_path)

    df = yf.download(tickers=ticker, period=period, interval="1h", progress=False)
    if df is None or df.empty:
        raise ValueError(f"Yahoo Finance returned empty H1 data for {ticker} ({period})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "time"})
    closed_df, _, _, _ = split_closed_and_live(df, timeframe="1h", now_utc=datetime.now(timezone.utc))
    return _finalize_ohlcv_dataframe(closed_df)


def _to_utc_epoch_seconds(raw_time):
    if isinstance(raw_time, datetime):
        dt = raw_time if raw_time.tzinfo is not None else raw_time.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    if hasattr(raw_time, "timestamp"):
        return float(raw_time.timestamp())
    parsed = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
    dt = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def _seconds_before_latest(time_col):
    if hasattr(time_col, "dt"):
        utc_series = pd.to_datetime(time_col, utc=True)
        age_sec = -(utc_series - utc_series.iloc[-1]).dt.total_seconds().to_numpy(dtype=np.float64)
    else:
        raw_vals = getattr(time_col, "_vals", None)
        items = raw_vals if raw_vals is not None else list(time_col)
        epochs = np.asarray([_to_utc_epoch_seconds(v) for v in items], dtype=np.float64)
        age_sec = epochs[-1] - epochs
    if not np.isfinite(age_sec).all():
        raise ValueError("Candle timestamps contain NaT or invalid values")
    return age_sec


def _extract_col_matrix(df, cols, dtype=np.float32):
    sub = df[list(cols)]
    if hasattr(sub, "values"):
        return np.asarray(sub.values, dtype=dtype)
    return np.column_stack([np.asarray(df[c], dtype=dtype) for c in cols])


def _validate_feature_matrix(values):
    if not np.isfinite(values).all():
        raise ValueError("OHLCV dataset contains NaN or Inf values")
    if (values[:, :4] <= 0).any():
        raise ValueError("OHLC prices must be strictly positive")
    if (values[:, 1] < values[:, 2]).any():
        raise ValueError("Corrupted candle detected with high < low")


def validate_recent_candles(df, val_days=30, min_bars=300):
    if df is None or len(df) == 0:
        raise ValueError("Empty candle dataframe")
    _validate_feature_matrix(_extract_col_matrix(df, PRICE_COLS, dtype=np.float64))
    age_sec = _seconds_before_latest(df["time"])
    if (np.diff(age_sec) > 0).any():
        raise ValueError("Candle timestamps are not sorted in ascending order")
    recent_count = int(np.sum(age_sec <= float(val_days) * 86400.0))
    if recent_count < int(min_bars):
        raise ValueError(f"Insufficient recent candles in last {val_days}d: got {recent_count}, need >= {min_bars}")
    return recent_count


def compute_recency_weights(num_windows, min_weight=0.5, max_weight=1.5):
    count = int(num_windows)
    if count <= 0:
        raise ValueError("num_windows must be positive")
    if count == 1:
        return np.ones(1, dtype=np.float64)
    weights = np.linspace(float(min_weight), float(max_weight), count, dtype=np.float64)
    return weights / np.mean(weights)


def _extract_normalized_windows(values, stamps, indices, context_len, window_len):
    clip_bound = 5.0
    x_list, s_list = [], []
    for start in indices:
        win = values[start : start + window_len]
        ctx = win[:context_len]
        mean = np.mean(ctx, axis=0, keepdims=True)
        std = np.std(ctx, axis=0, keepdims=True)
        normed = np.clip((win - mean) / (std + 1e-5), -clip_bound, clip_bound).astype(np.float32)
        x_list.append(normed)
        s_list.append(stamps[start : start + window_len])
    return np.stack(x_list, axis=0), np.stack(s_list, axis=0)


def _resolve_split_bar(df, window_len, pred_len, step, val_days):
    total = len(df)
    min_val = int(pred_len) + 1 + step * 2
    fallback_split = total - max(int(total * DEFAULT_FALLBACK_VAL_RATIO), min_val)
    if val_days is None or float(val_days) <= 0:
        return fallback_split
    age_sec = _seconds_before_latest(df["time"])
    recent_mask = age_sec <= float(val_days) * 86400.0
    time_split = int(np.argmax(recent_mask)) if recent_mask.any() else fallback_split
    min_train = max(window_len + step * 10, int(total * 0.5))
    if time_split < min_train or (total - time_split) < min_val:
        return fallback_split
    return time_split


def build_kronos_datasets(df, context_len=200, pred_len=12, stride=4, val_days=30):
    window_len = int(context_len) + int(pred_len) + 1
    if len(df) < window_len * 2:
        raise ValueError(f"Need at least {window_len * 2} candles for train/val split, got {len(df)}")

    values = _extract_col_matrix(df, ALL_FEATURE_COLS, dtype=np.float32)
    _validate_feature_matrix(values)
    stamps = calc_time_stamps(pd.to_datetime(df["time"], utc=True)).values.astype(np.float32)

    step = max(int(stride), 1)
    split_bar = _resolve_split_bar(df, window_len, pred_len, step, val_days)
    train_limit = split_bar - window_len + 1
    val_start = max(0, split_bar - int(context_len))
    train_indices = list(range(0, max(train_limit, 0), step))
    val_indices = list(range(val_start, len(df) - window_len + 1, step))
    if not train_indices or not val_indices:
        raise ValueError("Insufficient candles to construct non-overlapping train and validation windows")

    train_x, train_s = _extract_normalized_windows(values, stamps, train_indices, context_len, window_len)
    val_x, val_s = _extract_normalized_windows(values, stamps, val_indices, context_len, window_len)
    return (train_x, train_s), (val_x, val_s)
