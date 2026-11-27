import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
import config

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "signals_history.json")
VN_TZ = timezone(timedelta(hours=7))


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


def _extract_direction(direction_str):
    text = str(direction_str or "")
    if "BUY" in text:
        return "BUY"
    if "SELL" in text:
        return "SELL"
    return ""


def _parse_utc_timestamp(val):
    if not val:
        return None
    if hasattr(val, "to_pydatetime"):
        val = val.to_pydatetime()
    if isinstance(val, datetime):
        dt = val if val.tzinfo is not None else val.replace(tzinfo=VN_TZ)
        return dt.astimezone(timezone.utc)
    s = str(val).strip()
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=VN_TZ)
        return dt.astimezone(timezone.utc)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(s[:len(fmt)], fmt).replace(tzinfo=VN_TZ).astimezone(timezone.utc)
            except ValueError:
                continue
    return None


def _signal_cutoff_utc(sig):
    candidates = [
        _parse_utc_timestamp(sig.get("open_time")),
        _parse_utc_timestamp(sig.get("open_time_vn")),
    ]
    valid = [dt for dt in candidates if dt is not None]
    return max(valid) if valid else None


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


def parse_signal_time_vn(sig):
    """Chuyển đổi thời gian mở của tín hiệu về datetime theo giờ Việt Nam (UTC+7)."""
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

    last_sig = open_signals[-1]
    first_entry = float(last_sig.get("entry_price", entry_price))
    first_side = _extract_direction(last_sig.get("direction", ""))
    min_dist = getattr(config, "MIN_ENTRY_DISTANCE_ATR", 0.8)
    atr_val = float(atr) if (atr is not None and atr > 0) else 10.0

    if first_side == "BUY" and new_side == "BUY":
        req = first_entry - (atr_val * min_dist)
        if entry_price > req:
            return f"Đang có 1 lệnh BUY ({first_entry}). Entry mới ({entry_price}) chưa tối ưu hơn mức yêu cầu (cần <= {req:.2f})."
        print(f"🎯 [{symbol}] Điểm Entry mới ({entry_price}) tối ưu hơn lệnh 1 ({first_entry}) -> Mở lệnh thứ 2 riêng biệt!")
    elif first_side == "SELL" and new_side == "SELL":
        req = first_entry + (atr_val * min_dist)
        if entry_price < req:
            return f"Đang có 1 lệnh SELL ({first_entry}). Entry mới ({entry_price}) chưa tối ưu hơn mức yêu cầu (cần >= {req:.2f})."
        print(f"🎯 [{symbol}] Điểm Entry mới ({entry_price}) tối ưu hơn lệnh 1 ({first_entry}) -> Mở lệnh thứ 2 riêng biệt!")
    return None


def add_new_signal(symbol, direction, entry_price, tp, sl, time_str, timeframe="15m", atr=None):
    history = load_history()
    for sig in history:
        if sig.get("symbol") == symbol and str(sig.get("open_time")) == str(time_str):
            return False, f"Đã có tín hiệu cho cây nến này ({time_str})"

    open_signals = [s for s in history if s.get("symbol") == symbol and s.get("status") == "OPEN"]
    max_open = getattr(config, "MAX_OPEN_SIGNALS_PER_SYMBOL", 2)
    if len(open_signals) >= max_open:
        return False, f"Đã đạt tối đa {max_open} lệnh đang mở cho cặp {symbol}. Chờ chốt TP/SL mới mở tiếp."
    if open_signals:
        err = _validate_entry_distance(symbol, open_signals, direction, entry_price, atr)
        if err:
            return False, err

    now_vn = datetime.now(VN_TZ)
    signal_id = f"{symbol}_{now_vn.strftime('%Y%m%d_%H%M%S')}"
    history.append({
        "id": signal_id, "symbol": symbol, "timeframe": timeframe, "direction": direction,
        "entry_price": entry_price, "tp": tp, "sl": sl, "open_time": time_str,
        "open_time_vn": now_vn.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "OPEN", "closed_at": None, "close_price": None, "order_num": len(open_signals) + 1,
    })
    save_history(history)
    return True, signal_id


def _apply_hit(sig, status, close_price, closed_at):
    sig["status"] = status
    sig["closed_at"] = closed_at
    sig["close_price"] = close_price
    label = "chạm TP" if status == "HIT_TP" else "dính SL"
    icon = "🎯" if status == "HIT_TP" else "🛑"
    print(f"{icon} [{sig['symbol']}] Tín hiệu {_extract_direction(sig['direction'])} {label} tại giá {close_price}! (Nến {closed_at})")


def update_open_signals(symbol, current_high, current_low, current_price, current_time_str, df=None):
    """Kiểm tra các tín hiệu OPEN xem đã chạm TP hay SL chưa."""
    history = load_history()
    rows = _iter_ohlc_rows(df)
    cur_utc = _parse_utc_timestamp(current_time_str)
    updated = False

    for sig in history:
        if sig.get("symbol") != symbol or sig.get("status") != "OPEN":
            continue
        cutoff_utc = _signal_cutoff_utc(sig)
        if rows:
            for row_t, h, l in rows:
                row_utc = _parse_utc_timestamp(row_t)
                if row_utc is None or (cutoff_utc is not None and row_utc <= cutoff_utc):
                    continue
                hit = _evaluate_bar_hit(sig["direction"], h, l, sig["tp"], sig["sl"])
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


def _filter_signals_in_range(history, start_vn, end_vn):
    matched = []
    for sig in history:
        t_vn = parse_signal_time_vn(sig)
        if t_vn and start_vn <= t_vn <= end_vn:
            matched.append(sig)
    return matched


def _count_symbol_stats(signals, symbol):
    tp = sum(1 for s in signals if s["symbol"] == symbol and s["status"] == "HIT_TP")
    sl = sum(1 for s in signals if s["symbol"] == symbol and s["status"] == "HIT_SL")
    decided = tp + sl
    return tp, sl, (tp / decided * 100) if decided > 0 else 0.0


def _summarize_window(start_date, end_date):
    start_vn = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0, tzinfo=VN_TZ)
    end_vn = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=VN_TZ)
    items = _filter_signals_in_range(load_history(), start_vn, end_vn)
    hit_tp = sum(1 for s in items if s["status"] == "HIT_TP")
    hit_sl = sum(1 for s in items if s["status"] == "HIT_SL")
    running = sum(1 for s in items if s["status"] == "OPEN")
    win_rate = (hit_tp / (hit_tp + hit_sl) * 100) if (hit_tp + hit_sl) > 0 else 0.0
    return len(items), hit_tp, hit_sl, running, win_rate, _count_symbol_stats(items, "XAUUSD"), _count_symbol_stats(items, "BTCUSD")


def generate_daily_report(target_date=None):
    """Tạo báo cáo tổng hợp hiệu suất của ngày hôm qua (chốt lúc 06:00 sáng)."""
    if target_date is None:
        target_date = (datetime.now(VN_TZ) - timedelta(days=1)).date()
    elif isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()

    total, hit_tp, hit_sl, running, win_rate, (gold_tp, gold_sl, _), (btc_tp, btc_sl, _) = _summarize_window(target_date, target_date)
    return (
        "📊 *[VÀNG Đánh BTC] - BÁO CÁO HIỆU SUẤT NGÀY*\n"
        f"📅 *Ngày tổng kết:* `{target_date.strftime('%d/%m/%Y')}`\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 *Tổng số tín hiệu phát:* `{total}`\n"
        f"🎯 *Đạt mục tiêu (TP):* `{hit_tp}` lệnh\n"
        f"🛑 *Cắt lỗ (SL):* `{hit_sl}` lệnh\n"
        f"⏳ *Đang chạy (Running):* `{running}` lệnh\n"
        f"🏆 *Tỷ lệ thắng (Win Rate):* *{win_rate:.1f}%*\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "*Chi tiết theo sản phẩm:*\n"
        f"🟡 *Vàng (XAUUSD - H1):* `{gold_tp}` TP / `{gold_sl}` SL\n"
        f"🟠 *Bitcoin (BTCUSD - M15):* `{btc_tp}` TP / `{btc_sl}` SL\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "💡 _Hệ thống tự động tiếp tục quét chu kỳ mới ngày hôm nay._"
    )


def _resolve_week_bounds(ref_date):
    weekday = ref_date.weekday()
    if weekday == 5:
        return ref_date - timedelta(days=5), ref_date - timedelta(days=1)
    if weekday == 6:
        return ref_date - timedelta(days=6), ref_date - timedelta(days=2)
    monday = ref_date - timedelta(days=weekday)
    return monday, monday + timedelta(days=4)


def generate_weekly_report(ref_date=None):
    """Tạo báo cáo tổng hợp hiệu suất tuần (Thứ 2 đến Thứ 6)."""
    if ref_date is None:
        ref_date = datetime.now(VN_TZ).date()
    elif isinstance(ref_date, str):
        ref_date = datetime.strptime(ref_date, "%Y-%m-%d").date()

    monday_date, friday_date = _resolve_week_bounds(ref_date)
    total, hit_tp, hit_sl, running, win_rate, (gold_tp, gold_sl, gold_win), (btc_tp, btc_sl, btc_win) = _summarize_window(monday_date, friday_date)
    return (
        "🏆 *[VÀNG Đánh BTC] - BÁO CÁO TỔNG KẾT TUẦN*\n"
        f"📅 *Giai đoạn:* Từ {monday_date.strftime('%d/%m/%Y')} đến {friday_date.strftime('%d/%m/%Y')}\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 *Tổng số tín hiệu cả tuần:* `{total}`\n"
        f"🎯 *Đạt mục tiêu (TP):* `{hit_tp}` lệnh\n"
        f"🛑 *Cắt lỗ (SL):* `{hit_sl}` lệnh\n"
        f"⏳ *Đang chạy (Running):* `{running}` lệnh\n"
        f"🔥 *TỶ LỆ THẮNG TUẦN (WIN RATE):* *{win_rate:.1f}%*\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "*Chi tiết theo sản phẩm:*\n"
        f"🟡 *Vàng (XAUUSD - H1):* `{gold_tp}` TP / `{gold_sl}` SL (Win: *{gold_win:.1f}%*)\n"
        f"🟠 *Bitcoin (BTCUSD - M15):* `{btc_tp}` TP / `{btc_sl}` SL (Win: *{btc_win:.1f}%*)\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🎉 _Chúc anh em cuối tuần nghỉ ngơi vui vẻ! Hẹn gặp lại vào tuần tới._"
    )
