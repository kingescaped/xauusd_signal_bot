import time
import schedule
import argparse
import os
import config
from fetch_data import get_gold_candles
from kronos_predictor import predict_future
from chart_visualizer import create_signal_chart
from telegram_notifier import send_telegram_photo, send_telegram_message, test_telegram

def run_job():
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Đang kiểm tra dữ liệu nến Vàng ({config.SYMBOL})...")
    
    # 1. Kéo dữ liệu nến
    df = get_gold_candles()
    if df is None or len(df) < 30:
        print("[LỖI] Dữ liệu nến không đủ hoặc không tải được.")
        return

    # 2. Phân tích & dự báo xu hướng
    result = predict_future(df)
    print(f"-> Kết quả: {result['direction']} | Độ tin cậy: {result['confidence']}%")

    # 3. Luôn tạo ảnh biểu đồ trực quan kèm các mốc TP / SL
    chart_path = os.path.join(os.path.dirname(__file__), "signal_chart.png")
    created_chart = create_signal_chart(
        df=df,
        predicted_direction=result['direction'],
        current_price=result['current_price'],
        tp=result['tp'],
        sl=result['sl'],
        output_path=chart_path
    )

    # 4. Kiểm tra điều kiện ngưỡng tin cậy để gửi thông báo
    if result['confidence'] >= config.CONFIDENCE_THRESHOLD and "SIDEWAY" not in result['direction']:
        caption = (
            "🔔 *[KRONOS AI] - CẢNH BÁO TÍN HIỆU VÀNG (XAUUSD)*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 *Sản phẩm:* `{config.SYMBOL}` | Khung: `{config.TIMEFRAME}`\n"
            f"💰 *Giá hiện tại:* `${result['current_price']}`\n"
            f"🎯 *Tín hiệu:* *{result['direction']}*\n"
            f"📈 *Độ tin cậy:* `{result['confidence']}%`\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🚀 *Mục tiêu (TP):* `${result['tp']}`\n"
            f"🛑 *Cắt lỗ (SL):* `${result['sl']}`\n"
            f"⚡ *Biến động ATR:* `{result['atr']}` giá\n\n"
            "⚠️ _Lưu ý: Tín hiệu tham khảo hỗ trợ phân tích, luôn quản lý vốn an toàn!_"
        )

        if created_chart and os.path.exists(created_chart):
            send_telegram_photo(created_chart, caption)
        else:
            send_telegram_message(caption)
    else:
        print(f"-> Độ tin cậy {result['confidence']}% chưa đạt ngưỡng {config.CONFIDENCE_THRESHOLD}% hoặc thị trường đang Sideway. Bỏ qua lượt này.")

def main():
    parser = argparse.ArgumentParser(description="Kronos Gold Telegram Signal Bot")
    parser.add_argument("--test", action="store_true", help="Kiểm tra kết nối Telegram vào nhóm")
    parser.add_argument("--once", action="store_true", help="Chạy quét tín hiệu 1 lần duy nhất ngay bây giờ (dành cho GitHub Actions)")
    args = parser.parse_args()

    if args.test:
        print("Đang gửi tin nhắn thử nghiệm vào nhóm Telegram...")
        test_telegram()
        return

    if args.once:
        print("Chạy quét tín hiệu 1 lần (Single Run Mode)...")
        run_job()
        return

    print("==================================================")
    print("   KRONOS GOLD TELEGRAM SIGNAL BOT ĐÃ KHỞI ĐỘNG   ")
    print(f"   - Theo dõi: {config.SYMBOL} ({config.TIMEFRAME})")
    print(f"   - Nhóm Telegram: {config.TELEGRAM_GROUP_ID}")
    print("==================================================")
    
    # Chạy lần đầu tiên
    run_job()

    # Lên lịch chạy định kỳ mỗi tiếng 1 lần
    schedule.every(1).hours.at(":01").do(run_job)

    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == "__main__":
    main()
