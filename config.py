import os

# Cấu hình Bot Telegram (Hỗ trợ cả biến môi trường từ GitHub Secrets hoặc cấu hình trực tiếp)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "9193848531:AAGphQqE0tP_SGxiiBMRUaZUL0umU_L14yY")
TELEGRAM_GROUP_ID = os.getenv("TELEGRAM_GROUP_ID", "-1002144845536")

# Cấu hình thị trường
SYMBOL = "GC=F"       # Mã Vàng thế giới trên Yahoo Finance (Gold Futures)
TIMEFRAME = "1h"      # Khung thời gian: '15m', '1h', '4h', '1d'
CANDLE_LIMIT = 100    # Số nến lấy về để đưa vào mô hình

# Ngưỡng tin cậy tối thiểu để bắn tín hiệu (70%)
CONFIDENCE_THRESHOLD = 70.0
