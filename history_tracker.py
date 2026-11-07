import json
import os
from datetime import datetime, timedelta, timezone

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "signals_history.json")

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

def add_new_signal(symbol, direction, entry_price, tp, sl, time_str):
    history = load_history()
    # Kiểm tra xem có tín hiệu mở cùng cặp trong 30 phút gần nhất không (tránh bắn lặp nến kế tiếp)
    for sig in reversed(history[-10:]):
        if sig['symbol'] == symbol and sig['status'] == 'OPEN':
            return False, "Đã có tín hiệu đang mở cho cặp này"
            
    signal_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    new_sig = {
        "id": signal_id,
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry_price,
        "tp": tp,
        "sl": sl,
        "open_time": time_str,
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

def generate_daily_report():
    """Tạo báo cáo tổng hợp hiệu suất trong 24 giờ qua."""
    history = load_history()
    now_vn = datetime.now(timezone(timedelta(hours=7)))
    yesterday_vn = now_vn - timedelta(hours=24)
    yesterday_str = yesterday_vn.strftime('%Y-%m-%d %H:%M')

    recent_signals = []
    for sig in history:
        # Lọc tín hiệu mở trong 24 giờ qua
        try:
            sig_time = datetime.strptime(sig['open_time'][:16], '%Y-%m-%d %H:%M')
            if sig_time >= yesterday_vn.replace(tzinfo=None):
                recent_signals.append(sig)
        except:
            recent_signals.append(sig)

    total = len(recent_signals)
    hit_tp = sum(1 for s in recent_signals if s['status'] == 'HIT_TP')
    hit_sl = sum(1 for s in recent_signals if s['status'] == 'HIT_SL')
    running = sum(1 for s in recent_signals if s['status'] == 'OPEN')
    
    decided = hit_tp + hit_sl
    win_rate = (hit_tp / decided * 100) if decided > 0 else 0.0

    # Phân loại theo mã
    gold_tp = sum(1 for s in recent_signals if s['symbol'] == 'XAUUSD' and s['status'] == 'HIT_TP')
    gold_sl = sum(1 for s in recent_signals if s['symbol'] == 'XAUUSD' and s['status'] == 'HIT_SL')
    btc_tp = sum(1 for s in recent_signals if s['symbol'] == 'BTCUSD' and s['status'] == 'HIT_TP')
    btc_sl = sum(1 for s in recent_signals if s['symbol'] == 'BTCUSD' and s['status'] == 'HIT_SL')

    report_msg = (
        "📊 *[KRONOS AI] - BÁO CÁO TỔNG HỢP 24H (06:00 SÁNG)*\n"
        f"📅 _Ngày báo cáo: {now_vn.strftime('%d/%m/%Y')}_\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 *Tổng số tín hiệu phát:* `{total}`\n"
        f"🎯 *Đạt mục tiêu (TP):* `{hit_tp}` lệnh\n"
        f"🛑 *Cắt lỗ (SL):* `{hit_sl}` lệnh\n"
        f"⏳ *Đang chạy (Running):* `{running}` lệnh\n"
        f"🏆 *Tỷ lệ thắng (Win Rate):* *{win_rate:.1f}%*\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "*Chi tiết theo sản phẩm:*\n"
        f"🟡 *Vàng (XAUUSD):* `{gold_tp}` TP / `{gold_sl}` SL\n"
        f"🟠 *Bitcoin (BTCUSD):* `{btc_tp}` TP / `{btc_sl}` SL\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "💡 _Hệ thống tự động reset thống kê và tiếp tục quét nến M15._"
    )
    return report_msg
