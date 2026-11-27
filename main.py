import argparse
import html
import os
import time
from datetime import datetime, timedelta, timezone
import schedule
import config
from bot_state import has_sent_report, mark_report_sent
from chart_visualizer import create_signal_chart
from fetch_data import get_candles, split_closed_and_live
from history_tracker import add_new_signal, generate_daily_report, generate_weekly_report, update_open_signals
from kronos_predictor import predict_future
from telegram_notifier import send_telegram_message, send_telegram_photo, test_telegram

VN_TZ = timezone(timedelta(hours=7))


def is_market_open(symbol_name, now_vn):
    """Kiểm tra thị trường có đang mở cửa không (Crypto 24/7, Vàng nghỉ cuối tuần & 05-06h VN)."""
    if symbol_name != "XAUUSD":
        return True, "Thị trường Crypto mở cửa 24/7"

    weekday, hour = now_vn.weekday(), now_vn.hour
    if weekday == 5 and hour >= 5:
        return False, "Thị trường Vàng đã đóng cửa cuối tuần (từ 05:00 sáng Thứ Bảy). Sẽ mở lại lúc 06:00 sáng Thứ Hai tới!"
    if weekday == 6:
        return False, "Hôm nay là Chủ Nhật: Thị trường Vàng đang đóng cửa cuối tuần. Sẽ mở lại lúc 06:00 sáng Thứ Hai tới!"
    if weekday == 0 and hour < 6:
        return False, "Thị trường Vàng chưa mở cửa tuần mới (Mở lúc 06:00 sáng Thứ Hai VN)."
    if weekday in (1, 2, 3, 4) and hour == 5:
        return False, "Giờ nghỉ kỹ thuật hằng ngày của Vàng thế giới (05:00 - 06:00 sáng VN)."
    return True, "Thị trường Vàng đang mở cửa giao dịch bình thường"


def check_and_send_daily_report_if_due(now_vn=None):
    """Kiểm tra nếu đang trong khoảng 06:00 - 06:14 sáng (giờ VN) thì gửi báo cáo đúng 1 lần."""
    now_vn = now_vn or datetime.now(VN_TZ)
    if now_vn.hour != config.DAILY_REPORT_HOUR_VN or now_vn.minute >= 15:
        return

    today_key = now_vn.strftime("%Y-%m-%d")
    if not has_sent_report("daily_report", today_key):
        print(f"[{now_vn.strftime('%H:%M:%S')}] Đến 6:00 sáng VN -> Đang tổng hợp báo cáo ngày hôm qua...")
        if send_telegram_message(generate_daily_report()):
            mark_report_sent("daily_report", today_key)
            print("Đã gửi báo cáo ngày hôm qua vào nhóm!")

    if now_vn.weekday() == 5 and not has_sent_report("weekly_report", today_key):
        print(f"[{now_vn.strftime('%H:%M:%S')}] Sáng Thứ Bảy -> Đang tổng hợp báo cáo tổng kết tuần...")
        if send_telegram_message(generate_weekly_report()):
            mark_report_sent("weekly_report", today_key)
            print("Đã gửi báo cáo tổng kết tuần vào nhóm!")


def _dispatch_telegram(chart_path, caption):
    if chart_path and os.path.exists(chart_path):
        send_telegram_photo(chart_path, caption)
    else:
        send_telegram_message(caption)


def _build_signal_caption(item, result, tf_label):
    nhan_dinh = "Nhận định của Vàng" if item["name"] == "XAUUSD" else "Nhận định của BTC"
    digits = int(item.get("digits", 2))
    adx_info = f" | <b>ADX:</b> <code>{result.get('adx', 0):.1f}</code>" if "adx" in result else ""
    agree_info = f" | Đồng thuận: <code>{result['sample_agreement']}%</code>" if "sample_agreement" in result else ""
    unit = html.escape(str(item["unit"]))
    return (
        f"🎯 <b>[VÀNG Đánh BTC] - TÍN HIỆU VÀO LỆNH {html.escape(item['label'])}</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Sản phẩm:</b> <code>{html.escape(item['name'])}</code> | <b>Khung:</b> <code>{html.escape(tf_label)}</code>\n"
        f"💰 <b>Giá vào (Entry):</b> <code>{result['current_price']:.{digits}f} {unit}</code>\n"
        f"🚀 <b>Tín hiệu:</b> <b>{html.escape(result['direction'])}</b>\n"
        f"📈 <b>Độ tin cậy:</b> <code>{result['confidence']}%</code> (Đạt chuẩn ≥ {config.CONFIDENCE_THRESHOLD}%){agree_info}\n"
        f"🔮 <b>Kỳ vọng:</b> <code>{result['pct_change']:+.2f}%</code> ({item['predict_len']} nến tới)\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>Mục tiêu (TP):</b> <code>{result['tp']:.{digits}f} {unit}</code>\n"
        f"🛑 <b>Cắt lỗ (SL):</b> <code>{result['sl']:.{digits}f} {unit}</code>\n"
        f"⚡ <b>Biến động ATR:</b> <code>{result['atr']}</code>{adx_info}\n\n"
        f"💡 <b>{nhan_dinh}:</b> Tín hiệu theo đà xu hướng đang đẹp, kỳ vọng đạt mục tiêu TP!\n\n"
        "⚠️ <i>Hệ thống [VÀNG Đánh BTC] tự động theo dõi TP/SL và tổng kết lúc 06:00 sáng!</i>"
    )


def _build_force_caption(item, result, tf_label):
    nhan_dinh = "Nhận định của Vàng" if item["name"] == "XAUUSD" else "Nhận định của BTC"
    digits = int(item.get("digits", 2))
    unit = html.escape(str(item["unit"]))
    ai_note = "" if result.get("ai_predicted", False) else "\n⚠️ <b>[CẢNH BÁO: Kronos AI không khả dụng, không tạo tín hiệu giao dịch]</b>"
    return (
        f"📊 <b>[VÀNG Đánh BTC] - CẬP NHẬT PHÂN TÍCH {html.escape(item['label'])}</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Sản phẩm:</b> <code>{html.escape(item['name'])}</code> | <b>Khung:</b> <code>{html.escape(tf_label)}</code>\n"
        f"💰 <b>Giá hiện tại:</b> <code>{result['current_price']:.{digits}f} {unit}</code>\n"
        f"🎯 <b>Trạng thái:</b> <b>{html.escape(result['direction'])}</b>\n"
        f"📈 <b>Độ tin cậy:</b> <code>{result['confidence']}%</code> <i>(Cần ≥ {config.CONFIDENCE_THRESHOLD}% để vào lệnh)</i>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 <b>{nhan_dinh}:</b> Thị trường đang Sideway / đi ngang tích lũy, chưa xuất hiện điểm vào lệnh có xác suất thắng cao. Tiếp tục quan sát!{ai_note}"
    )


def _handle_symbol_result(item, df, time_str, result, manual_force):
    symbol_name, label, timeframe = item["name"], item["label"], item["timeframe"]
    ai_ok = bool(result.get("ai_predicted", False))
    is_high_conf = ai_ok and result["confidence"] >= config.CONFIDENCE_THRESHOLD and "SIDEWAY" not in result["direction"]
    if not is_high_conf and not manual_force:
        msg = "Kronos AI không khả dụng -> bỏ qua tín hiệu." if not ai_ok else f"Độ tin cậy {result['confidence']}% (< {config.CONFIDENCE_THRESHOLD}%) hoặc Sideway -> Giữ yên lặng tuyệt đối, không gửi tin nhắn."
        print(f"-> {symbol_name}: {msg}")
        return

    tf_label = "M15" if timeframe.lower() == "15m" else "H1"
    chart_path = os.path.join(os.path.dirname(__file__), f"chart_{symbol_name.lower()}.png")
    create_signal_chart(df=df, symbol_name=symbol_name, symbol_label=label, timeframe=timeframe, unit=item["unit"], predicted_direction=result["direction"], current_price=result["current_price"], tp=result["tp"], sl=result["sl"], output_path=chart_path, digits=int(item.get("digits", 2)))

    if is_high_conf:
        added, reason = add_new_signal(symbol=symbol_name, direction=result["direction"], entry_price=result["current_price"], tp=result["tp"], sl=result["sl"], time_str=time_str, timeframe=timeframe, atr=result.get("atr", 1.0), metadata=result)
        if added:
            print(f"-> {symbol_name}: 🚀 Bắn tín hiệu vào nhóm Telegram thành công! (Mã: {reason})")
            _dispatch_telegram(chart_path, _build_signal_caption(item, result, tf_label))
        else:
            print(f"-> {symbol_name}: ⏸️ {reason}")
        return

    print(f"-> {symbol_name}: [Có cờ --force] Gửi ảnh phân tích Sideway.")
    _dispatch_telegram(chart_path, _build_force_caption(item, result, tf_label))


def _process_symbol(item, now_vn, manual_force):
    symbol_name, ticker, label = item["name"], item["ticker"], item["label"]
    timeframe, pred_len, period = item["timeframe"], item["predict_len"], item["period"]

    market_open, market_msg = is_market_open(symbol_name, now_vn)
    if not market_open:
        print(f"[{symbol_name}] ⏸️ {market_msg} -> Tạm dừng quét & giữ yên lặng tuyệt đối, không gửi tin nhắn.")
        return

    if timeframe.lower() == "1h" and now_vn.minute >= 15 and not manual_force:
        print(f"[{symbol_name} - 1H] Chưa đóng nến 1H (phút {now_vn.minute}) -> Chờ đến đầu tiếng tiếp theo.")
        df_check = get_candles(ticker, timeframe="15m", period="2d", limit=24)
        if df_check is not None and len(df_check) > 0:
            last_c = df_check.iloc[-1]
            update_open_signals(symbol_name, float(last_c["high"]), float(last_c["low"]), float(last_c["close"]), str(last_c["time"]), df=df_check)
        return

    print(f"\n--- Đang xử lý: {label} (Khung {timeframe.upper()}) ---")
    fetch_limit = getattr(config, "INDICATOR_CANDLE_LIMIT", config.CANDLE_LIMIT) + 1
    df = get_candles(ticker, timeframe=timeframe, period=period, limit=fetch_limit)
    if df is None or len(df) == 0:
        print(f"[BỎ QUA] Không đủ nến cho {label}.")
        return

    latest = df.iloc[-1]
    update_open_signals(symbol_name, float(latest["high"]), float(latest["low"]), float(latest["close"]), str(latest["time"]), df=df)

    closed_df, live_price, _, had_forming = split_closed_and_live(
        df, timeframe=timeframe, now_utc=now_vn.astimezone(timezone.utc),
    )
    if closed_df is None or len(closed_df) < 25:
        print(f"[BỎ QUA] Không đủ nến đã đóng cho {label}.")
        return
    if had_forming:
        print(f"-> [{symbol_name}] Đã tách nến đang chạy, AI đọc {len(closed_df)} nến đã đóng (Entry live: {live_price}).")

    closed_time_str = str(closed_df.iloc[-1]["time"])
    result = predict_future(closed_df, symbol_name, timeframe=timeframe, pred_len=pred_len, live_price=live_price)
    print(f"-> Phân tích {symbol_name} ({timeframe}): {result['direction']} | Độ tin cậy: {result['confidence']}%")
    _handle_symbol_result(item, closed_df, closed_time_str, result, manual_force)


def run_single_cycle(manual_force=False, now_vn=None):
    now_vn = now_vn or datetime.now(VN_TZ)
    print(f"\n[{now_vn.strftime('%Y-%m-%d %H:%M:%S')} VN] Bắt đầu chu kỳ quét nến...")
    if manual_force:
        print("⚡ [CHẾ ĐỘ ÉP BUỘC] Luôn gửi ảnh và phân tích kể cả khi Sideway.")
    else:
        print(f"🤖 [CHẠY TỰ ĐỘNG] Giữ im lặng khi Sideway/cuối tuần. Chỉ gửi tin nhắn khi có kèo đẹp độ tin cậy >= {config.CONFIDENCE_THRESHOLD}%.")

    check_and_send_daily_report_if_due(now_vn=now_vn)
    for item in config.SYMBOLS:
        _process_symbol(item, now_vn, manual_force)


def main():
    parser = argparse.ArgumentParser(description="VÀNG Đánh BTC Signal Bot")
    parser.add_argument("--test", action="store_true", help="Kiểm tra kết nối Telegram vào nhóm")
    parser.add_argument("--report", action="store_true", help="Gửi ngay báo cáo hiệu suất ngày hôm qua vào nhóm")
    parser.add_argument("--report-week", action="store_true", help="Gửi ngay báo cáo tổng kết tuần (Thứ 2 - Thứ 6) vào nhóm")
    parser.add_argument("--once", action="store_true", help="Chạy quét 1 chu kỳ ngay bây giờ")
    parser.add_argument("--force", action="store_true", help="Ép buộc gửi ảnh biểu đồ kể cả lúc Sideway (cho chế độ bấm tay)")
    args = parser.parse_args()

    if args.test:
        test_telegram()
        return
    if args.report:
        send_telegram_message(generate_daily_report())
        return
    if args.report_week:
        send_telegram_message(generate_weekly_report())
        return
    if args.once:
        run_single_cycle(manual_force=args.force)
        return

    print("==================================================")
    print("   [VÀNG Đánh BTC] BOT: BTC (M15) & VÀNG (1H)     ")
    print("   - BTCUSD: Khung M15 (quét mỗi 15 phút)         ")
    print("   - XAUUSD: Khung H1  (quét đầu mỗi tiếng)       ")
    print("   - Báo cáo định kỳ: 06:00 Sáng hàng ngày        ")
    print("==================================================")
    run_single_cycle(manual_force=False)
    schedule.every(15).minutes.do(lambda: run_single_cycle(manual_force=False))
    while True:
        schedule.run_pending()
        time.sleep(15)


if __name__ == "__main__":
    main()
