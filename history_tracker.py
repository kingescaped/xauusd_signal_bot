import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
import config
from market_time import VN_TZ, to_utc_datetime
from report_builder import build_daily_report, build_weekly_report

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "signals_history.json")

_METADATA_KEYS = (
    "confidence", "pct_change", "pred_end_price", "pred_high", "pred_low",
    "ref_close", "atr", "adx", "ema_50", "ema_200",
    "sample_agreement", "sample_std_pct", "forecast_window_vn", "strategy_version",
)


def _symbol_digits_and_atr(symbol):
    for item in getattr(config, "SYMBOLS", []):
        if item.get("name") == symbol:
            return int(item.get("digits", 2)), float(item.get("atr_fallback", 10.0))
    return 2, (5.0 if symbol == "XAUUSD" else 150.0)


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupted history file {HISTORY_FILE}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Invalid history format in {HISTORY_FILE}: expected list")
    return data


def save_history(history):
    history_dir = os.path.dirname(os.path.abspath(HISTORY_FILE))
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=history_dir,
            prefix=".signals_history_", suffix=".tmp", delete=False,
        ) as tmp:
            tmp_path = tmp.name
            json.dump(history, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, HISTORY_FILE)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def remove_signal(signal_id):
    if not signal_id:
        return False
    history = load_history()
    filtered = [s for s in history if s.get("id") != signal_id]
    if len(filtered) == len(history):
        return False
    save_history(filtered)
    return True


def has_open_signals(symbol, recent_minutes=None, now_utc=None):
    history = load_history()
    open_items = [s for s in history if s.get("symbol") == symbol and s.get("status") == "OPEN"]
    if not open_items or recent_minutes is None:
        return bool(open_items)
    ref_utc = to_utc_datetime(now_utc) if now_utc is not None else datetime.now(timezone.utc)
    threshold = ref_utc - timedelta(minutes=recent_minutes)
    return any((_signal_cutoff_utc(s) or ref_utc) >= threshold for s in open_items)


def _extract_direction(direction_str):
    text = str(direction_str or "")
    if "BUY" in text:
        return "BUY"
    if "SELL" in text:
        return "SELL"
    return ""


def _parse_utc_timestamp(val):
    return to_utc_datetime(val, default_tz=VN_TZ)


def _signal_cutoff_utc(sig):
    candidates = [
        _parse_utc_timestamp(sig.get("open_time")),
        _parse_utc_timestamp(sig.get("open_time_vn")),
    ]
    valid = [dt for dt in candidates if dt is not None]
    return max(valid) if valid else None


def parse_signal_time_vn(sig):
    for field in ("open_time_vn", "open_time"):
        dt = _parse_utc_timestamp(sig.get(field))
        if dt is not None:
            return dt.astimezone(VN_TZ)
    parts = sig.get("id", "").split("_")
    if len(parts) >= 3:
        try:
            return datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y%m%d_%H%M%S").replace(tzinfo=VN_TZ)
        except ValueError:
            pass
    return None


def _validate_entry_distance(symbol, open_signals, direction, entry_price, atr):
    new_side = _extract_direction(direction)
    allow_hedge = getattr(config, "ALLOW_OPPOSITE_HEDGING", False)
    if not allow_hedge and new_side:
        for sig in open_signals:
            existing_side = _extract_direction(sig.get("direction", ""))
            if existing_side and existing_side != new_side:
                return f"Đang có lệnh {existing_side} mở cho {symbol}, chặn mở lệnh {new_side} ngược chiều."

    digits, default_atr = _symbol_digits_and_atr(symbol)
    last_sig = open_signals[-1]
    first_entry = float(last_sig.get("entry_price", entry_price))
    first_side = _extract_direction(last_sig.get("direction", ""))
    min_dist = getattr(config, "MIN_ENTRY_DISTANCE_ATR", 0.8)
    atr_val = float(atr) if (atr is not None and atr > 0) else default_atr

    if first_side == "BUY" and new_side == "BUY":
        req = first_entry - (atr_val * min_dist)
        if entry_price > req:
            return f"Đang có 1 lệnh BUY ({first_entry:.{digits}f}). Entry DCA #2 ({entry_price:.{digits}f}) cần <= {req:.{digits}f}."
        print(f"🎯 [{symbol}] Entry DCA #2 ({entry_price:.{digits}f}) tối ưu hơn lệnh 1 ({first_entry:.{digits}f}) -> Mở lệnh DCA thứ 2!")
    elif first_side == "SELL" and new_side == "SELL":
        req = first_entry + (atr_val * min_dist)
        if entry_price < req:
            return f"Đang có 1 lệnh SELL ({first_entry:.{digits}f}). Entry DCA #2 ({entry_price:.{digits}f}) cần >= {req:.{digits}f}."
        print(f"🎯 [{symbol}] Entry DCA #2 ({entry_price:.{digits}f}) tối ưu hơn lệnh 1 ({first_entry:.{digits}f}) -> Mở lệnh DCA thứ 2!")
    return None


def _extract_signal_metadata(metadata):
    if not isinstance(metadata, dict):
        return {}
    return {k: metadata[k] for k in _METADATA_KEYS if k in metadata and metadata[k] is not None}


def _same_candle_time(existing_time, new_time):
    t1, t2 = _parse_utc_timestamp(existing_time), _parse_utc_timestamp(new_time)
    if t1 is not None and t2 is not None:
        return t1 == t2
    return str(existing_time) == str(new_time)


def add_new_signal(symbol, direction, entry_price, tp, sl, time_str, timeframe="15m", atr=None, metadata=None):
    history = load_history()
    for sig in history:
        if sig.get("symbol") == symbol and _same_candle_time(sig.get("open_time"), time_str):
            return False, f"Đã có tín hiệu cho cây nến này ({time_str})"

    open_signals = [s for s in history if s.get("symbol") == symbol and s.get("status") == "OPEN"]
    max_open = getattr(config, "MAX_OPEN_SIGNALS_PER_SYMBOL", 2)
    if len(open_signals) >= max_open:
        return False, f"Đã đạt tối đa {max_open} lệnh DCA đang mở cho cặp {symbol}. Chờ chốt TP/SL mới mở tiếp."
    if open_signals:
        err = _validate_entry_distance(symbol, open_signals, direction, entry_price, atr)
        if err:
            return False, err

    now_vn = datetime.now(VN_TZ)
    order_num = len(open_signals) + 1
    base_id = f"{symbol}_{now_vn.strftime('%Y%m%d_%H%M%S')}"
    existing_ids = {s.get("id") for s in history}
    signal_id = f"{base_id}_{order_num}" if (order_num > 1 or base_id in existing_ids) else base_id
    new_sig = {
        "id": signal_id, "symbol": symbol, "timeframe": timeframe, "direction": direction,
        "entry_price": entry_price, "tp": tp, "sl": sl, "open_time": time_str,
        "open_time_vn": now_vn.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "OPEN", "closed_at": None, "close_price": None,
        "order_num": order_num, "is_dca": order_num > 1,
    }
    new_sig.update(_extract_signal_metadata(metadata))
    history.append(new_sig)
    save_history(history)
    return True, signal_id


def _evaluate_bar_hit(direction_str, high, low, tp, sl):
    side = _extract_direction(direction_str)
    if side == "BUY":
        if low <= sl:
            return "HIT_SL", sl
        if high >= tp:
            return "HIT_TP", tp
    elif side == "SELL":
        if high >= sl:
            return "HIT_SL", sl
        if low <= tp:
            return "HIT_TP", tp
    return None


def _iter_ohlc_rows(df):
    if df is None or getattr(df, "empty", True):
        return []
    cols = list(df.columns)
    if "high" not in cols or "low" not in cols:
        return []
    time_col = "time" if "time" in cols else cols[0]
    return [(row[time_col], float(row["high"]), float(row["low"])) for _, row in df.iterrows()]


def _apply_hit(sig, status, close_price, closed_at):
    sig["status"] = status
    sig["closed_at"] = closed_at
    closed_dt = _parse_utc_timestamp(closed_at)
    if closed_dt is not None:
        sig["closed_at_vn"] = closed_dt.astimezone(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")
    sig["close_price"] = close_price
    label = "chạm TP" if status == "HIT_TP" else "dính SL"
    icon = "🎯" if status == "HIT_TP" else "🛑"
    print(f"{icon} [{sig['symbol']}] Tín hiệu {_extract_direction(sig['direction'])} {label} tại giá {close_price}! (Nến {closed_at})")


def update_open_signals(symbol, current_high, current_low, current_price, current_time_str, df=None):
    history = load_history()
    rows = _iter_ohlc_rows(df)
    cur_utc = _parse_utc_timestamp(current_time_str)
    updated = False

    for sig in history:
        if sig.get("symbol") != symbol or sig.get("status") != "OPEN":
            continue
        cutoff_utc = _signal_cutoff_utc(sig)
        if rows:
            for row_t, high_val, low_val in rows:
                row_utc = _parse_utc_timestamp(row_t)
                if row_utc is None or (cutoff_utc is not None and row_utc <= cutoff_utc):
                    continue
                hit = _evaluate_bar_hit(sig["direction"], high_val, low_val, sig["tp"], sig["sl"])
                if hit:
                    _apply_hit(sig, hit[0], hit[1], str(row_t))
                    updated = True
                    break
        elif cur_utc is not None and (cutoff_utc is None or cur_utc > cutoff_utc):
            hit = _evaluate_bar_hit(sig["direction"], current_high, current_low, sig["tp"], sig["sl"])
            if hit:
                _apply_hit(sig, hit[0], hit[1], current_time_str)
                updated = True

    if updated:
        save_history(history)
    return updated


def generate_daily_report(target_date=None):
    return build_daily_report(load_history(), parse_signal_time_vn, target_date=target_date)


def generate_weekly_report(ref_date=None):
    return build_weekly_report(load_history(), parse_signal_time_vn, ref_date=ref_date)
