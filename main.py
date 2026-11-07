import time
import schedule
import argparse
import os
from datetime import datetime, timezone, timedelta
import config
from fetch_data import get_candles
from kronos_predictor import predict_future
from chart_visualizer import create_signal_chart
from history_tracker import add_new_signal, update_open_signals, generate_daily_report
from telegram_notifier import send_telegram_photo, send_telegram_message, test_telegram

def check_and_send_daily_report_if_due():
    """Kiểm tra nếu đang trong khoảng 06:00 - 06:15 sáng (giờ VN) thì gửi báo cáo 24h."""
    now_vn = datetime.now(timezone(timedelta(hours=7)))
    # Kiểm tra đúng 6 giờ sáng
    if now_vn.hour == config.DAILY_REPORT_HOUR_VN and now_vn.minute < 16:
        print(f"[{now_vn.strftime('%H:%M:%S')}] Đến 6:00 sáng VN -> Đang tổng hợp báo cáo 24h...")
        report_text = generate_daily_report()
        send_telegram_message(report_text)
        print("Đã gửi báo cáo tổng kết 24h vào nhóm!")

def run_single_cycle():
    now_vn = datetime.now(timezone(timedelta(hours=7)))
    print(f"\n[{now_vn.strftime('%Y-%m-%d %H:%M:%S')} VN] Bắt đầu quét tín hiệu M15...")

    # 1. Kiểm tra gửi báo cáo 6:00 sáng
    check_and_send_daily_report_if_due()

    # 2. Quét lần lượt từng sản phẩm (Vàng, Bitcoin)
    for item in config.SYMBOLS:
        symbol_name = item['name']
        ticker = item['ticker']
        label = item['label']
        unit = item['unit']

        print(f"\n--- Đang xử lý: {label} ({ticker}) ---")
        df = get_candles(ticker, timeframe=config.TIMEFRAME)
        if df is None or len(df) < 25:
            print(f"[BỎ QUA] Không đủ nến cho {label}.")
            continue

        latest_candle = df.iloc[-1]
        current_time_str = str(latest_candle['time'])
        current_high = float(latest_candle['high'])
        current_low = float(latest_candle['low'])
        current_close = float(latest_candle['close'])

        # Cập nhật xem các lệnh đang chạy trước đó có chạm TP/SL ở cây nến này không
        update_open_signals(symbol_name, current_high, current_low, current_close, current_time_str)

        # Phân tích dự báo cây nến tiếp theo
        result = predict_future(df, symbol_name)
        print(f"-> Phân tích {symbol_name}: {result['direction']} | Độ tin cậy: {result['confidence']}%")

        # Kiểm tra điều kiện bắn tín hiệu
        if result['confidence'] >= config.CONFIDENCE_THRESHOLD and "SIDEWAY" not in result['direction']:
            # Lưu vào lịch sử (nếu chưa có lệnh cùng chiều đang mở)
            added, reason = add_new_signal(
                symbol=symbol_name,
                direction=result['direction'],
                entry_price=result['current_price'],
                tp=result['tp'],
                sl=result['sl'],
                time_str=current_time_str
            )

            if added:
                # Vẽ biểu đồ riêng cho sản phẩm này
                chart_filename = f"chart_{symbol_name.lower()}.png"
                chart_path = os.path.join(os.path.dirname(__file__), chart_filename)
                create_signal_chart(
                    df=df,
                    symbol_name=symbol_name,
                    symbol_label=label,
                    unit=unit,
                    predicted_direction=result['direction'],
                    current_price=result['current_price'],
                    tp=result['tp'],
                    sl=result['sl'],
                    output_path=chart_path
                )

                # Soạn tin nhắn Telegram
                caption = (
                    f"🔔 *[KRONOS AI] - CẢNH BÁO TÍN HIỆU {label}*\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 *Sản phẩm:* `{symbol_name}` | Khung: `{config.TIMEFRAME}`\n"
                    f"💰 *Giá vào (Entry):* `{result['current_price']} {unit}`\n"
                    f"🎯 *Tín hiệu:* *{result['direction']}*\n"
                    f"📈 *Độ tin cậy:* `{result['confidence']}%`\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"🚀 *Mục tiêu (TP):* `{result['tp']} {unit}`\n"
                    f"🛑 *Cắt lỗ (SL):* `{result['sl']} {unit}`\n"
                    f"⚡ *Biến động ATR:* `{result['atr']}`\n\n"
                    "⚠️ _Tín hiệu tự động được lưu để theo dõi TP/SL và báo cáo lúc 06:00 sáng!_"
                )

                if os.path.exists(chart_path):
                    send_telegram_photo(chart_path, caption)
                else:
                    send_telegram_message(caption)
            else:
                print(f"-> {symbol_name}: {reason}")
        else:
            print(f"-> {symbol_name}: Độ tin cậy {result['confidence']}% chưa đạt {config.CONFIDENCE_THRESHOLD}% hoặc đang Sideway.")

def main():
    parser = argparse.ArgumentParser(description="Kronos Gold & BTC Telegram Signal Bot")
    parser.add_argument("--test", action="store_true", help="Kiểm tra kết nối Telegram vào nhóm")
    parser.add_argument("--report", action="store_true", help="Gửi ngay báo cáo tổng hợp 24h vào nhóm")
    parser.add_argument("--once", action="store_true", help="Chạy quét 1 lần duy nhất ngay bây giờ (cho GitHub Actions)")
    args = parser.parse_args()

    if args.test:
        print("Đang gửi tin nhắn thử nghiệm vào nhóm Telegram...")
        test_telegram()
        return

    if args.report:
        print("Đang gửi báo cáo hiệu suất 24h...")
        report_text = generate_daily_report()
        send_telegram_message(report_text)
        return

    if args.once:
        print("Chạy quét 1 chu kỳ (Single Run Mode)...")
        run_single_cycle()
        return

    print("==================================================")
    print("   KRONOS GOLD & BITCOIN BOT ĐÃ KHỞI ĐỘNG (M15)   ")
    print(f"   - Theo dõi: XAUUSD & BTCUSD ({config.TIMEFRAME})")
    print(f"   - Nhóm Telegram: {config.TELEGRAM_GROUP_ID}")
    print(f"   - Báo cáo định kỳ: 06:00 Sáng hàng ngày")
    print("==================================================")
    
    # Chạy lần đầu ngay khi bật
    run_single_cycle()

    # Lên lịch chạy mỗi 15 phút
    schedule.every(15).minutes.do(run_single_cycle)

    while True:
        schedule.run_pending()
        time.sleep(15)

if __name__ == "__main__":
    main()
