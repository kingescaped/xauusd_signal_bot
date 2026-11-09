import time
import schedule
import argparse
import os
from datetime import datetime, timezone, timedelta
import config
from fetch_data import get_candles
from kronos_predictor import predict_future
from chart_visualizer import create_signal_chart
from history_tracker import add_new_signal, update_open_signals, generate_daily_report, generate_weekly_report
from telegram_notifier import send_telegram_photo, send_telegram_message, test_telegram

def is_market_open(symbol_name, now_vn):
    """
    Kiểm tra thị trường có đang mở cửa không:
    - BTCUSD: Crypto mở cửa 24/7 không nghỉ.
    - XAUUSD (Vàng thế giới):
        + Đóng cửa cuối tuần: Từ 05:00 sáng Thứ Bảy đến 06:00 sáng Thứ Hai (giờ VN).
        + Nghỉ bảo trì kỹ thuật hàng ngày: 05:00 - 06:00 sáng từ Thứ Ba đến Thứ Sáu.
    """
    if symbol_name != "XAUUSD":
        return True, "Thị trường Crypto mở cửa 24/7"

    weekday = now_vn.weekday()  # 0: Thứ 2, 1: Thứ 3, ..., 5: Thứ 7, 6: Chủ Nhật
    hour = now_vn.hour

    if weekday == 5 and hour >= 5:
        return False, "Thị trường Vàng đã đóng cửa cuối tuần (từ 05:00 sáng Thứ Bảy). Sẽ mở lại lúc 06:00 sáng Thứ Hai tới!"
    if weekday == 6:
        return False, "Hôm nay là Chủ Nhật: Thị trường Vàng đang đóng cửa cuối tuần. Sẽ mở lại lúc 06:00 sáng Thứ Hai tới!"
    if weekday == 0 and hour < 6:
        return False, "Thị trường Vàng chưa mở cửa tuần mới (Mở lúc 06:00 sáng Thứ Hai VN)."
    if weekday in [1, 2, 3, 4] and hour == 5:
        return False, "Giờ nghỉ kỹ thuật hằng ngày của Vàng thế giới (05:00 - 06:00 sáng VN)."

    return True, "Thị trường Vàng đang mở cửa giao dịch bình thường"

def check_and_send_daily_report_if_due():
    """Kiểm tra nếu đang trong khoảng 06:00 - 06:15 sáng (giờ VN) thì gửi báo cáo."""
    now_vn = datetime.now(timezone(timedelta(hours=7)))
    if now_vn.hour == config.DAILY_REPORT_HOUR_VN and now_vn.minute < 16:
        print(f"[{now_vn.strftime('%H:%M:%S')}] Đến 6:00 sáng VN -> Đang tổng hợp báo cáo ngày hôm qua...")
        daily_text = generate_daily_report()
        send_telegram_message(daily_text)
        print("Đã gửi báo cáo ngày hôm qua vào nhóm!")

        # Nếu là sáng Thứ Bảy (weekday == 5) -> Gửi thêm báo cáo tổng kết tuần
        if now_vn.weekday() == 5:
            print(f"[{now_vn.strftime('%H:%M:%S')}] Sáng Thứ Bảy -> Đang tổng hợp báo cáo tổng kết tuần...")
            weekly_text = generate_weekly_report()
            send_telegram_message(weekly_text)
            print("Đã gửi báo cáo tổng kết tuần vào nhóm!")

def run_single_cycle(manual_force=False):
    now_vn = datetime.now(timezone(timedelta(hours=7)))
    print(f"\n[{now_vn.strftime('%Y-%m-%d %H:%M:%S')} VN] Bắt đầu chu kỳ quét nến...")
    if manual_force:
        print("⚡ [CHẾ ĐỘ ÉP BUỘC] Luôn gửi ảnh và phân tích kể cả khi Sideway.")
    else:
        print("🤖 [CHẠY TỰ ĐỘNG] Giữ im lặng khi Sideway/cuối tuần. Chỉ gửi tin nhắn khi có kèo đẹp độ tin cậy >= 70%.")

    # 1. Kiểm tra gửi báo cáo 6:00 sáng
    check_and_send_daily_report_if_due()

    # 2. Quét lần lượt từng sản phẩm theo khung thời gian riêng:
    # - BTCUSD: Quét mỗi 15 phút (M15)
    # - XAUUSD: Quét theo nến 1 Giờ (H1 - vào đầu mỗi tiếng)
    for item in config.SYMBOLS:
        symbol_name = item['name']
        ticker = item['ticker']
        label = item['label']
        timeframe = item['timeframe']
        unit = item['unit']
        pred_len = item['predict_len']
        period = item['period']

        # KIỂM TRA GIỜ MỞ/ĐÓNG CỬA THỊ TRƯỜNG (Áp dụng riêng cho Vàng XAUUSD)
        # Cuối tuần hoặc ngoài giờ giao dịch: giữ im lặng tuyệt đối, không gửi tin nhắn
        market_open, market_status_msg = is_market_open(symbol_name, now_vn)
        if not market_open:
            print(f"[{symbol_name}] ⏸️ {market_status_msg} -> Tạm dừng quét & giữ yên lặng tuyệt đối, không gửi tin nhắn.")
            continue

        # XỬ LÝ KHUNG NẾN 1H:
        is_top_of_hour = now_vn.minute < 16
        if timeframe.lower() == "1h" and not is_top_of_hour and not manual_force:
            print(f"[{symbol_name} - 1H] Chưa đóng nến 1H (phút {now_vn.minute}) -> Chờ đến đầu tiếng tiếp theo.")
            df_check = get_candles(ticker, timeframe="15m", period="2d", limit=5)
            if df_check is not None and len(df_check) > 0:
                last_c = df_check.iloc[-1]
                update_open_signals(symbol_name, float(last_c['high']), float(last_c['low']), float(last_c['close']), str(last_c['time']), df=df_check)
            continue

        print(f"\n--- Đang xử lý: {label} (Khung {timeframe.upper()}) ---")
        df = get_candles(ticker, timeframe=timeframe, period=period)
        if df is None or len(df) < 25:
            print(f"[BỎ QUA] Không đủ nến cho {label}.")
            continue

        latest_candle = df.iloc[-1]
        current_time_str = str(latest_candle['time'])
        current_high = float(latest_candle['high'])
        current_low = float(latest_candle['low'])
        current_close = float(latest_candle['close'])

        # Cập nhật xem các lệnh trước đó có chạm TP/SL không
        update_open_signals(symbol_name, current_high, current_low, current_close, current_time_str, df=df)

        # Phân tích dự báo bằng Kronos-base
        result = predict_future(df, symbol_name, timeframe=timeframe, pred_len=pred_len)
        print(f"-> Phân tích {symbol_name} ({timeframe}): {result['direction']} | Độ tin cậy: {result['confidence']}%")

        tf_label = "M15" if timeframe.lower() == "15m" else "H1"
        is_high_confidence = (result['confidence'] >= config.CONFIDENCE_THRESHOLD and "SIDEWAY" not in result['direction'])
        
        # Tiêu đề nhận định riêng theo từng sản phẩm
        nhan_dinh_title = "Nhận định của Vàng" if symbol_name == "XAUUSD" else "Nhận định của BTC"

        # Luôn vẽ ảnh biểu đồ nến với tiêu đề LVQ Std SIGNAL
        chart_filename = f"chart_{symbol_name.lower()}.png"
        chart_path = os.path.join(os.path.dirname(__file__), chart_filename)
        create_signal_chart(
            df=df,
            symbol_name=symbol_name,
            symbol_label=label,
            timeframe=timeframe,
            unit=unit,
            predicted_direction=result['direction'],
            current_price=result['current_price'],
            tp=result['tp'],
            sl=result['sl'],
            output_path=chart_path
        )

        # =========================================================================
        # TRƯỜNG HỢP 1: Có kèo đẹp (Độ tin cậy >= 70% và không Sideway)
        # =========================================================================
        if is_high_confidence:
            added, reason = add_new_signal(
                symbol=symbol_name,
                direction=result['direction'],
                entry_price=result['current_price'],
                tp=result['tp'],
                sl=result['sl'],
                time_str=current_time_str,
                timeframe=timeframe
            )

            if added:
                print(f"-> {symbol_name}: 🚀 Bắn tín hiệu vào nhóm Telegram thành công! (Mã: {reason})")
                caption = (
                    f"🎯 *[VÀNG Đánh BTC] - TÍN HIỆU VÀO LỆNH {label}*\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 *Sản phẩm:* `{symbol_name}` | *Khung:* `{tf_label}`\n"
                    f"💰 *Giá vào (Entry):* `{result['current_price']} {unit}`\n"
                    f"🚀 *Tín hiệu:* *{result['direction']}*\n"
                    f"📈 *Độ tin cậy:* `{result['confidence']}%` (Đạt chuẩn ≥ 70%)\n"
                    f"🔮 *Kỳ vọng:* `{result['pct_change']:+.2f}%` ({pred_len} nến tới)\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"🎯 *Mục tiêu (TP):* `{result['tp']} {unit}`\n"
                    f"🛑 *Cắt lỗ (SL):* `{result['sl']} {unit}`\n"
                    f"⚡ *Biến động ATR:* `{result['atr']}`\n\n"
                    f"💡 *{nhan_dinh_title}:* Tín hiệu theo đà xu hướng đang đẹp, kỳ vọng đạt mục tiêu TP!\n\n"
                    "⚠️ _Hệ thống [VÀNG Đánh BTC] tự động theo dõi TP/SL và tổng kết lúc 06:00 sáng!_"
                )

                if os.path.exists(chart_path):
                    send_telegram_photo(chart_path, caption)
                else:
                    send_telegram_message(caption)
            else:
                print(f"-> {symbol_name}: ⏸️ {reason}")

        # =========================================================================
        # TRƯỜNG HỢP 2: Bấm tay có cờ --force (chỉ dùng khi muốn test thủ công)
        # =========================================================================
        elif manual_force:
            print(f"-> {symbol_name}: [Có cờ --force] Gửi ảnh phân tích Sideway.")
            caption = (
                f"📊 *[VÀNG Đánh BTC] - CẬP NHẬT PHÂN TÍCH {label}*\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 *Sản phẩm:* `{symbol_name}` | *Khung:* `{tf_label}`\n"
                f"💰 *Giá hiện tại:* `{result['current_price']} {unit}`\n"
                f"🎯 *Trạng thái:* *{result['direction']}*\n"
                f"📈 *Độ tin cậy:* `{result['confidence']}%` *(Cần ≥ 70% để vào lệnh)*\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 *{nhan_dinh_title}:* Thị trường đang Sideway / đi ngang tích lũy, chưa xuất hiện điểm vào lệnh có xác suất thắng cao. Tiếp tục quan sát!"
            )

            if os.path.exists(chart_path):
                send_telegram_photo(chart_path, caption)
            else:
                send_telegram_message(caption)

        # =========================================================================
        # TRƯỜNG HỢP 3: Thị trường Sideway hoặc độ tin cậy < 70% khi chạy tự động
        # -> Giữ yên lặng tuyệt đối, KHÔNG gửi tin nhắn làm phiền nhóm
        # =========================================================================
        else:
            print(f"-> {symbol_name}: Độ tin cậy {result['confidence']}% (< {config.CONFIDENCE_THRESHOLD}%) hoặc Sideway -> Giữ yên lặng tuyệt đối, không gửi tin nhắn.")

def main():
    parser = argparse.ArgumentParser(description="VÀNG Đánh BTC Signal Bot")
    parser.add_argument("--test", action="store_true", help="Kiểm tra kết nối Telegram vào nhóm")
    parser.add_argument("--report", action="store_true", help="Gửi ngay báo cáo hiệu suất ngày hôm qua vào nhóm")
    parser.add_argument("--report-week", action="store_true", help="Gửi ngay báo cáo tổng kết tuần (Thứ 2 - Thứ 6) vào nhóm")
    parser.add_argument("--once", action="store_true", help="Chạy quét 1 chu kỳ ngay bây giờ")
    parser.add_argument("--force", action="store_true", help="Ép buộc gửi ảnh biểu đồ kể cả lúc Sideway (cho chế độ bấm tay)")
    args = parser.parse_args()

    if args.test:
        print("Đang gửi tin nhắn thử nghiệm vào nhóm Telegram...")
        test_telegram()
        return

    if args.report:
        print("Đang gửi báo cáo hiệu suất ngày hôm qua...")
        report_text = generate_daily_report()
        send_telegram_message(report_text)
        return

    if args.report_week:
        print("Đang gửi báo cáo tổng kết tuần...")
        report_text = generate_weekly_report()
        send_telegram_message(report_text)
        return

    is_force = args.force

    if args.once:
        print(f"Chạy quét 1 chu kỳ (Single Run Mode | Chế độ ép buộc gửi tin nhắn: {is_force})...")
        run_single_cycle(manual_force=is_force)
        return

    print("==================================================")
    print("   [VÀNG Đánh BTC] BOT: BTC (M15) & VÀNG (1H)     ")
    print(f"   - BTCUSD: Khung M15 (quét mỗi 15 phút)         ")
    print(f"   - XAUUSD: Khung H1  (quét đầu mỗi tiếng)       ")
    print(f"   - Báo cáo định kỳ: 06:00 Sáng hàng ngày        ")
    print("==================================================")
    
    run_single_cycle(manual_force=False)
    schedule.every(15).minutes.do(lambda: run_single_cycle(manual_force=False))

    while True:
        schedule.run_pending()
        time.sleep(15)

if __name__ == "__main__":
    main()
