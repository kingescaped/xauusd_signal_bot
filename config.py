import os

# Cấu hình Bot Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "9193848531:AAGphQqE0tP_SGxiiBMRUaZUL0umU_L14yY")
TELEGRAM_GROUP_ID = os.getenv("TELEGRAM_GROUP_ID", "-1002144845536")

# Danh sách các cặp tài sản theo dõi
SYMBOLS = [
    {
        "name": "XAUUSD",
        "ticker": "GC=F",         # Vàng thế giới
        "label": "VÀNG (XAUUSD)",
        "unit": "$/oz"
    },
    {
        "name": "BTCUSD",
        "ticker": "BTC-USD",      # Bitcoin
        "label": "BITCOIN (BTCUSD)",
        "unit": "$"
    }
]

# Khung thời gian quét tín hiệu
TIMEFRAME = "15m"      # 15 phút
CANDLE_LIMIT = 60      # 60 nến gần nhất (đủ cho M15)

# Ngưỡng tin cậy tối thiểu để bắn tín hiệu (70%)
CONFIDENCE_THRESHOLD = 70.0

# Giờ gửi báo cáo tổng kết hàng ngày (Giờ Việt Nam UTC+7)
DAILY_REPORT_HOUR_VN = 6  # 06:00 sáng
