import json
import os
from datetime import datetime, timedelta, timezone

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "signals_history.json")
VN_TZ = timezone(timedelta(hours=7))

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[CẢNH BÁO] Không đọc được file lịch sử: {e}")
        return []

def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[LỖI] Không lưu được file lịch sử: {e}")

def parse_signal_time_vn(sig):
    """
    Chuyển đổi thời gian mở của tín hiệu về datetime theo giờ Việt Nam (UTC+7).
    """
    # 1. Ưu tiên open_time_vn nếu đã lưu sẵn
    if sig.get("open_time_vn"):
        try:
            dt = datetime.strptime(str(sig["open_time_vn"])[:19], "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=VN_TZ)
        except Exception:
            pass

    # 2. Xử lý open_time từ dữ liệu nến (có thể có tz offset như +00:00 hoặc -05:00)
    open_time_val = sig.get("open_time")
    if open_time_val:
        try:
            dt = datetime.fromisoformat(str(open_time_val))
            if dt.tzinfo is not None:
                return dt.astimezone(VN_TZ)
            else:
                return dt.replace(tzinfo=VN_TZ)
        except Exception:
            try:
                dt = datetime.strptime(str(open_time_val)[:16], "%Y-%m-%d %H:%M")
                return dt.replace(tzinfo=VN_TZ)
            except Exception:
                pass

    # 3. Fallback lấy từ id: SYMBOL_YYYYMMDD_HHMMSS
    sig_id = sig.get("id", "")
    try:
        parts = sig_id.split("_")
        if len(parts) >= 3:
            time_str = f"{parts[1]}_{parts[2]}"
            dt = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
            return dt.replace(tzinfo=VN_TZ)
    except Exception:
        pass

    return None

def add_new_signal(symbol, direction, entry_price, tp, sl, time_str, timeframe="15m"):
    history = load_history()
    # Kiểm tra xem có lệnh cùng cặp đang OPEN không (tránh spam lệnh liên tục)
    for sig in reversed(history[-10:]):
        if sig['symbol'] == symbol and sig['status'] == 'OPEN':
            return False, "Đã có tín hiệu đang mở cho cặp này"
            
    now_vn = datetime.now(VN_TZ)
    signal_id = f"{symbol}_{now_vn.strftime('%Y%m%d_%H%M%S')}"
    new_sig = {
        "id": signal_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "entry_price": entry_price,
        "tp": tp,
        "sl": sl,
        "open_time": time_str,
        "open_time_vn": now_vn.strftime('%Y-%m-%d %H:%M:%S'),
        "status": "OPEN",  # OPEN, HIT_TP, HIT_SL
        "closed_at": None,
        "close_price": None
    }
    history.append(new_sig)
    save_history(history)
    return True, signal_id

def update_open_signals(symbol, current_high, current_low, current_price, current_time_str):
    """Kiểm tra các tín hiệu OPEN xem đã chạm TP hay SL chưa."""
    history = load_history()
    updated = False
    
    for sig in history:
        if sig['symbol'] == symbol and sig['status'] == 'OPEN':
            dir_str = sig['direction']
            tp = sig['tp']
            sl = sig['sl']
            
            if "BUY" in dir_str:
                if current_high >= tp:
                    sig['status'] = "HIT_TP"
                    sig['closed_at'] = current_time_str
                    sig['close_price'] = tp
                    updated = True
                    print(f"🎯 [{symbol}] Tín hiệu BUY chạm TP tại giá {tp}!")
                elif current_low <= sl:
                    sig['status'] = "HIT_SL"
                    sig['closed_at'] = current_time_str
                    sig['close_price'] = sl
                    updated = True
                    print(f"🛑 [{symbol}] Tín hiệu BUY dính SL tại giá {sl}!")
            elif "SELL" in dir_str:
                if current_low <= tp:
                    sig['status'] = "HIT_TP"
                    sig['closed_at'] = current_time_str
                    sig['close_price'] = tp
                    updated = True
                    print(f"🎯 [{symbol}] Tín hiệu SELL chạm TP tại giá {tp}!")
                elif current_high >= sl:
                    sig['status'] = "HIT_SL"
                    sig['closed_at'] = current_time_str
                    sig['close_price'] = sl
                    updated = True
                    print(f"🛑 [{symbol}] Tín hiệu SELL dính SL tại giá {sl}!")

    if updated:
        save_history(history)
    return updated

def generate_daily_report(target_date=None):
    """
    Tạo báo cáo tổng hợp hiệu suất của ngày hôm qua (chốt lúc 06:00 sáng).
    Lọc toàn bộ tín hiệu được phát từ 00:00:00 đến 23:59:59 của ngày target_date theo giờ VN.
    """
    history = load_history()
    now_vn = datetime.now(VN_TZ)

    if target_date is None:
        target_date = (now_vn - timedelta(days=1)).date()
    elif isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()

    start_vn = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=VN_TZ)
    end_vn = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59, tzinfo=VN_TZ)

    target_date_str = target_date.strftime("%d/%m/%Y")

    daily_signals = []
    for sig in history:
        t_vn = parse_signal_time_vn(sig)
        if t_vn and start_vn <= t_vn <= end_vn:
            daily_signals.append(sig)

    total = len(daily_signals)
    hit_tp = sum(1 for s in daily_signals if s['status'] == 'HIT_TP')
    hit_sl = sum(1 for s in daily_signals if s['status'] == 'HIT_SL')
    running = sum(1 for s in daily_signals if s['status'] == 'OPEN')
    
    decided = hit_tp + hit_sl
    win_rate = (hit_tp / decided * 100) if decided > 0 else 0.0

    # Phân loại theo mã & khung thời gian
    gold_tp = sum(1 for s in daily_signals if s['symbol'] == 'XAUUSD' and s['status'] == 'HIT_TP')
    gold_sl = sum(1 for s in daily_signals if s['symbol'] == 'XAUUSD' and s['status'] == 'HIT_SL')
    btc_tp = sum(1 for s in daily_signals if s['symbol'] == 'BTCUSD' and s['status'] == 'HIT_TP')
    btc_sl = sum(1 for s in daily_signals if s['symbol'] == 'BTCUSD' and s['status'] == 'HIT_SL')

    report_msg = (
        "📊 *[VÀNG Đánh BTC] - BÁO CÁO HIỆU SUẤT NGÀY*\n"
        f"📅 *Ngày tổng kết:* `{target_date_str}`\n"
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
    return report_msg

def generate_weekly_report(ref_date=None):
    """
    Tạo báo cáo tổng hợp hiệu suất tuần (Thứ 2 đến Thứ 6).
    Mặc định chạy lúc 06:00 sáng Thứ Bảy để tổng kết tuần vừa qua.
    """
    history = load_history()
    now_vn = datetime.now(VN_TZ)

    if ref_date is None:
        ref_date = now_vn.date()
    elif isinstance(ref_date, str):
        ref_date = datetime.strptime(ref_date, "%Y-%m-%d").date()

    # Xác định Thứ Hai và Thứ Sáu của tuần
    weekday = ref_date.weekday()  # 0: Mon, ..., 4: Fri, 5: Sat, 6: Sun
    if weekday == 5:  # Thứ Bảy
        friday_date = ref_date - timedelta(days=1)
        monday_date = ref_date - timedelta(days=5)
    elif weekday == 6:  # Chủ Nhật
        friday_date = ref_date - timedelta(days=2)
        monday_date = ref_date - timedelta(days=6)
    else:
        # Nếu gọi giữa tuần: lấy từ Thứ Hai tuần này đến Thứ Sáu tuần này
        monday_date = ref_date - timedelta(days=weekday)
        friday_date = monday_date + timedelta(days=4)

    start_vn = datetime(monday_date.year, monday_date.month, monday_date.day, 0, 0, 0, tzinfo=VN_TZ)
    end_vn = datetime(friday_date.year, friday_date.month, friday_date.day, 23, 59, 59, tzinfo=VN_TZ)

    monday_str = monday_date.strftime("%d/%m/%Y")
    friday_str = friday_date.strftime("%d/%m/%Y")

    weekly_signals = []
    for sig in history:
        t_vn = parse_signal_time_vn(sig)
        if t_vn and start_vn <= t_vn <= end_vn:
            weekly_signals.append(sig)

    total = len(weekly_signals)
    hit_tp = sum(1 for s in weekly_signals if s['status'] == 'HIT_TP')
    hit_sl = sum(1 for s in weekly_signals if s['status'] == 'HIT_SL')
    running = sum(1 for s in weekly_signals if s['status'] == 'OPEN')

    decided = hit_tp + hit_sl
    win_rate = (hit_tp / decided * 100) if decided > 0 else 0.0

    # Phân loại theo mã & khung thời gian
    gold_tp = sum(1 for s in weekly_signals if s['symbol'] == 'XAUUSD' and s['status'] == 'HIT_TP')
    gold_sl = sum(1 for s in weekly_signals if s['symbol'] == 'XAUUSD' and s['status'] == 'HIT_SL')
    gold_decided = gold_tp + gold_sl
    gold_win = (gold_tp / gold_decided * 100) if gold_decided > 0 else 0.0

    btc_tp = sum(1 for s in weekly_signals if s['symbol'] == 'BTCUSD' and s['status'] == 'HIT_TP')
    btc_sl = sum(1 for s in weekly_signals if s['symbol'] == 'BTCUSD' and s['status'] == 'HIT_SL')
    btc_decided = btc_tp + btc_sl
    btc_win = (btc_tp / btc_decided * 100) if btc_decided > 0 else 0.0

    report_msg = (
        "🏆 *[VÀNG Đánh BTC] - BÁO CÁO TỔNG KẾT TUẦN*\n"
        f"📅 *Giai đoạn:* Từ {monday_str} đến {friday_str}\n"
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
    return report_msg
