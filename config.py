import os

# Cấu hình Bot Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_GROUP_ID = os.getenv("TELEGRAM_GROUP_ID", "")

# Cấu hình mô hình AI Kronos chính thức từ Hugging Face
KRONOS_MODEL_NAME = "NeoQuasar/Kronos-base"
KRONOS_TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
hf_token_val = os.getenv("HF_TOKEN", "").strip()
HF_TOKEN = hf_token_val if hf_token_val else None
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
else:
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

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
        "unit": "$",
        "predict_len": 6,         # Dự báo 6 nến 1H (6 tiếng tới)
        "period": "14d"
    }
]

CANDLE_LIMIT = 200     # Lấy 200 nến context cho mô hình Kronos

# Ngưỡng tin cậy tối thiểu để bắn tín hiệu (70%)
CONFIDENCE_THRESHOLD = 70.0

# Giờ gửi báo cáo tổng kết hàng ngày (Giờ Việt Nam UTC+7)
DAILY_REPORT_HOUR_VN = 6  # 06:00 sáng
