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
    if now_vn.hour == config.DAILY_REPORT_HOUR_VN and now_vn.minute < 16:
        print(f"[{now_vn.strftime('%H:%M:%S')}] Đến 6:00 sáng VN -> Đang tổng hợp báo cáo 24h...")
        report_text = generate_daily_report()
        send_telegram_message(report_text)
        print("Đã gửi báo cáo tổng kết 24h vào nhóm!")

def run_single_cycle(manual_force=False):
    now_vn = datetime.now(timezone(timedelta(hours=7)))
    print(f"\n[{now_vn.strftime('%Y-%m-%d %H:%M:%S')} VN] Bắt đầu chu kỳ quét nến...")

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

        # Nếu là nến 1H, chỉ quét phát tín hiệu mới vào đầu mỗi tiếng (phút 00 - 15) hoặc khi bấm nút chạy thủ công
        is_top_of_hour = now_vn.minute < 16
        if timeframe.lower() == "1h" and not is_top_of_hour and not manual_force:
            print(f"[{symbol_name} - 1H] Chưa đến giờ đóng nến 1H (hiện tại phút {now_vn.minute}). Bỏ qua quét tín hiệu mới.")
            # Vẫn kéo nến để kiểm tra trạng thái chạm TP/SL của các lệnh trước đó
            df_check = get_candles(ticker, timeframe="15m", period="2d", limit=5)
            if df_check is not None and len(df_check) > 0:
                last_c = df_check.iloc[-1]
                update_open_signals(symbol_name, float(last_c['high']), float(last_c['low']), float(last_c['close']), str(last_c['time']))
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

        # Cập nhật xem các lệnh đang chạy trước đó có chạm TP/SL không
        update_open_signals(symbol_name, current_high, current_low, current_close, current_time_str)

        # Phân tích dự báo bằng Kronos-base
        result = predict_future(df, symbol_name, timeframe=timeframe, pred_len=pred_len)
        print(f"-> Phân tích {symbol_name} ({timeframe}): {result['direction']} | Độ tin cậy: {result['confidence']}%")

        # Kiểm tra điều kiện bắn tín hiệu
        if result['confidence'] >= config.CONFIDENCE_THRESHOLD and "SIDEWAY" not in result['direction']:
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
                tf_label = "M15" if timeframe.lower() == "15m" else "H1"
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

                # Soạn tin nhắn Telegram
                caption = (
                    f"🔔 *[KRONOS-BASE AI] - TÍN HIỆU {label}*\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 *Sản phẩm:* `{symbol_name}` | *Khung:* `{tf_label}`\n"
                    f"💰 *Giá vào (Entry):* `{result['current_price']} {unit}`\n"
                    f"🎯 *Tín hiệu:* *{result['direction']}*\n"
                    f"📈 *Độ tin cậy:* `{result['confidence']}%`\n"
                    f"🔮 *Kỳ vọng:* `{result['pct_change']:+.2f}%` ({pred_len} nến tới)\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"🚀 *Mục tiêu (TP):* `{result['tp']} {unit}`\n"
                    f"🛑 *Cắt lỗ (SL):* `{result['sl']} {unit}`\n"
                    f"⚡ *Biến động ATR:* `{result['atr']}`\n\n"
                    "⚠️ _Mô hình AI Kronos-base tự động theo dõi TP/SL và báo cáo lúc 06:00 sáng!_"
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
    parser = argparse.ArgumentParser(description="Kronos Gold (1H) & BTC (M15) Signal Bot")
    parser.add_argument("--test", action="store_true", help="Kiểm tra kết nối Telegram vào nhóm")
    parser.add_argument("--report", action="store_true", help="Gửi ngay báo cáo tổng hợp 24h vào nhóm")
    parser.add_argument("--once", action="store_true", help="Chạy quét 1 chu kỳ ngay bây giờ (cho GitHub Actions)")
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
        run_single_cycle(manual_force=True)
        return

    print("==================================================")
    print("   KRONOS-BASE BOT: BTC (M15) & VÀNG (1H)         ")
    print(f"   - BTCUSD: Khung M15 (quét mỗi 15 phút)         ")
    print(f"   - XAUUSD: Khung H1  (quét đầu mỗi tiếng)       ")
    print(f"   - Báo cáo định kỳ: 06:00 Sáng hàng ngày        ")
    print("==================================================")
    
    run_single_cycle(manual_force=True)
    schedule.every(15).minutes.do(run_single_cycle)

    while True:
        schedule.run_pending()
        time.sleep(15)

if __name__ == "__main__":
    main()
