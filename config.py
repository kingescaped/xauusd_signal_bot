import os

# Cấu hình Bot Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "9193848531:AAGphQqE0tP_SGxiiBMRUaZUL0umU_L14yY")
TELEGRAM_GROUP_ID = os.getenv("TELEGRAM_GROUP_ID", "-1002144845536")

# Cấu hình mô hình AI Kronos chính thức từ Hugging Face
KRONOS_MODEL_NAME = "NeoQuasar/Kronos-base"
KRONOS_TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

# Danh sách cấu hình tài sản ĐA KHUNG THỜI GIAN
SYMBOLS = [
    {
        "name": "BTCUSD",
        "ticker": "BTC-USD",      # Bitcoin
        "label": "BITCOIN (BTCUSD)",
        "timeframe": "15m",       # Khung M15 cho BTC
        "unit": "$",
        "predict_len": 12,        # Dự báo 12 nến M15 (3 tiếng tới)
        "period": "5d"
    },
    {
        "name": "XAUUSD",
        "ticker": "GC=F",         # Vàng thế giới
        "label": "VÀNG (XAUUSD)",
        "timeframe": "1h",        # Khung 1H cho Vàng (ít nhiễu, chuẩn kỹ thuật)
        "unit": "$/oz",
        "predict_len": 6,         # Dự báo 6 nến 1H (6 tiếng tới)
        "period": "14d"
    }
]

CANDLE_LIMIT = 200     # Lấy 200 nến context cho mô hình Kronos

# Ngưỡng tin cậy tối thiểu để bắn tín hiệu (70%)
CONFIDENCE_THRESHOLD = 70.0

# Giờ gửi báo cáo tổng kết hàng ngày (Giờ Việt Nam UTC+7)
DAILY_REPORT_HOUR_VN = 6  # 06:00 sáng
