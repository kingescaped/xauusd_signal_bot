from datetime import datetime, timedelta
from market_time import VN_TZ, to_vn_datetime


def _filter_opened_in_range(history, start_vn, end_vn, parse_open_vn):
    matched = []
    for sig in history:
        t_vn = parse_open_vn(sig)
        if t_vn and start_vn <= t_vn <= end_vn:
            matched.append(sig)
    return matched


def _filter_carryover_closed_in_range(history, start_vn, end_vn, parse_open_vn):
    closed_carryover = []
    for sig in history:
        if sig.get("status") not in ("HIT_TP", "HIT_SL") or not sig.get("closed_at"):
            continue
        open_vn = parse_open_vn(sig)
        close_vn = to_vn_datetime(sig.get("closed_at"), default_tz=VN_TZ)
        if open_vn and close_vn and open_vn < start_vn and start_vn <= close_vn <= end_vn:
            closed_carryover.append(sig)
    return closed_carryover


def _count_symbol_stats(signals, symbol):
    tp = sum(1 for s in signals if s.get("symbol") == symbol and s.get("status") == "HIT_TP")
    sl = sum(1 for s in signals if s.get("symbol") == symbol and s.get("status") == "HIT_SL")
    decided = tp + sl
    return tp, sl, (tp / decided * 100.0) if decided > 0 else 0.0


def _summarize_window(history, start_date, end_date, parse_open_vn):
    start_vn = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0, tzinfo=VN_TZ)
    end_vn = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=VN_TZ)
    items = _filter_opened_in_range(history, start_vn, end_vn, parse_open_vn)
    carryover = _filter_carryover_closed_in_range(history, start_vn, end_vn, parse_open_vn)
    hit_tp = sum(1 for s in items if s.get("status") == "HIT_TP")
    hit_sl = sum(1 for s in items if s.get("status") == "HIT_SL")
    running = sum(1 for s in items if s.get("status") == "OPEN")
    win_rate = (hit_tp / (hit_tp + hit_sl) * 100.0) if (hit_tp + hit_sl) > 0 else 0.0
    return {
        "total": len(items),
        "hit_tp": hit_tp,
        "hit_sl": hit_sl,
        "running": running,
        "win_rate": win_rate,
        "gold": _count_symbol_stats(items, "XAUUSD"),
        "btc": _count_symbol_stats(items, "BTCUSD"),
        "carry_tp": sum(1 for s in carryover if s.get("status") == "HIT_TP"),
        "carry_sl": sum(1 for s in carryover if s.get("status") == "HIT_SL"),
    }


def _resolve_target_date(target_date, default_offset_days=0):
    if target_date is None:
        return (datetime.now(VN_TZ) - timedelta(days=default_offset_days)).date()
    if isinstance(target_date, str):
        return datetime.strptime(target_date, "%Y-%m-%d").date()
    return target_date


def build_daily_report(history, parse_open_vn, target_date=None):
    day = _resolve_target_date(target_date, default_offset_days=1)
    stats = _summarize_window(history, day, day, parse_open_vn)
    gold_tp, gold_sl, _ = stats["gold"]
    btc_tp, btc_sl, _ = stats["btc"]
    carry_line = ""
    if stats["carry_tp"] or stats["carry_sl"]:
        carry_line = f"\n📌 <b>Lệnh tồn ngày trước chốt hôm qua:</b> <code>{stats['carry_tp']}</code> TP / <code>{stats['carry_sl']}</code> SL"

    return (
        "📊 <b>[VÀNG Đánh BTC] - BÁO CÁO HIỆU SUẤT NGÀY</b>\n"
        f"📅 <b>Ngày tổng kết:</b> <code>{day.strftime('%d/%m/%Y')}</code>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Tổng số tín hiệu phát:</b> <code>{stats['total']}</code>\n"
        f"🎯 <b>Đạt mục tiêu (TP):</b> <code>{stats['hit_tp']}</code> lệnh\n"
        f"🛑 <b>Cắt lỗ (SL):</b> <code>{stats['hit_sl']}</code> lệnh\n"
        f"⏳ <b>Đang chạy (Running):</b> <code>{stats['running']}</code> lệnh\n"
        f"🏆 <b>Tỷ lệ thắng (Win Rate):</b> <b>{stats['win_rate']:.1f}%</b>{carry_line}\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>Chi tiết theo sản phẩm:</b>\n"
        f"🟡 <b>Vàng (XAUUSD - H1):</b> <code>{gold_tp}</code> TP / <code>{gold_sl}</code> SL\n"
        f"🟠 <b>Bitcoin (BTCUSD - M15):</b> <code>{btc_tp}</code> TP / <code>{btc_sl}</code> SL\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "💡 <i>Hệ thống tự động tiếp tục quét chu kỳ mới ngày hôm nay.</i>"
    )


def resolve_week_bounds(ref_date):
    weekday = ref_date.weekday()
    if weekday == 5:
        return ref_date - timedelta(days=5), ref_date - timedelta(days=1)
    if weekday == 6:
        return ref_date - timedelta(days=6), ref_date - timedelta(days=2)
    monday = ref_date - timedelta(days=weekday)
    return monday, monday + timedelta(days=4)


def build_weekly_report(history, parse_open_vn, ref_date=None):
    anchor = _resolve_target_date(ref_date, default_offset_days=0)
    monday_date, friday_date = resolve_week_bounds(anchor)
    stats = _summarize_window(history, monday_date, friday_date, parse_open_vn)
    gold_tp, gold_sl, gold_win = stats["gold"]
    btc_tp, btc_sl, btc_win = stats["btc"]
    return (
        "🏆 <b>[VÀNG Đánh BTC] - BÁO CÁO TỔNG KẾT TUẦN</b>\n"
        f"📅 <b>Giai đoạn:</b> Từ {monday_date.strftime('%d/%m/%Y')} đến {friday_date.strftime('%d/%m/%Y')}\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Tổng số tín hiệu cả tuần:</b> <code>{stats['total']}</code>\n"
        f"🎯 <b>Đạt mục tiêu (TP):</b> <code>{stats['hit_tp']}</code> lệnh\n"
        f"🛑 <b>Cắt lỗ (SL):</b> <code>{stats['hit_sl']}</code> lệnh\n"
        f"⏳ <b>Đang chạy (Running):</b> <code>{stats['running']}</code> lệnh\n"
        f"🔥 <b>TỶ LỆ THẮNG TUẦN (WIN RATE):</b> <b>{stats['win_rate']:.1f}%</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "<b>Chi tiết theo sản phẩm:</b>\n"
        f"🟡 <b>Vàng (XAUUSD - H1):</b> <code>{gold_tp}</code> TP / <code>{gold_sl}</code> SL (Win: <b>{gold_win:.1f}%</b>)\n"
        f"🟠 <b>Bitcoin (BTCUSD - M15):</b> <code>{btc_tp}</code> TP / <code>{btc_sl}</code> SL (Win: <b>{btc_win:.1f}%</b>)\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🎉 <i>Chúc anh em cuối tuần nghỉ ngơi vui vẻ! Hẹn gặp lại vào tuần tới.</i>"
    )
