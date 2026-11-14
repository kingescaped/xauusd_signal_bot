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

# Ngưỡng tin cậy tối thiểu để bắn tín hiệu (siết chặt từ 70% lên 75%)
CONFIDENCE_THRESHOLD = 75.0

# Ngưỡng biến động phần trăm tối thiểu trong chuỗi nến dự báo
BTC_THRESHOLD = 0.55       # Tối thiểu 0.55% cho BTC (M15) để loại trừ nhiễu nhỏ
GOLD_THRESHOLD = 0.30      # Tối thiểu 0.30% cho Vàng (1H)

# Cấu hình Quản lý Rủi ro TP / SL
ATR_SL_MULTIPLIER = 1.5    # SL = 1.5 * ATR (nới rộng chống quét râu nến M15 & H1)
MIN_RR_RATIO = 1.2         # Tỷ lệ R:R tối thiểu (TP >= 1.2 * khoảng cách SL để không bị giảm R:R)

# Cấu hình Quản lý Vị thế & Chặn nhồi lệnh
ALLOW_MULTIPLE_SIGNALS = False     # Bật chặn nhồi lệnh bừa bãi liên tục
MAX_OPEN_SIGNALS_PER_SYMBOL = 2    # Tối đa 2 lệnh riêng biệt cùng lúc cho mỗi cặp
MIN_ENTRY_DISTANCE_ATR = 0.8       # Lệnh thứ 2 chỉ được mở nếu Entry đẹp hơn ít nhất 0.8 * ATR

# Bộ lọc Kỹ thuật nâng cao
USE_TREND_FILTER = True            # Bộ lọc xu hướng EMA 50 & 200 (không đánh ngược sóng lớn)
USE_SIDEWAY_FILTER = True          # Bộ lọc thị trường Sideway bằng chỉ số ADX
ADX_THRESHOLD = 20.0               # ADX < 20 là thị trường đi ngang tích lũy, tạm dừng bắn lệnh

# Giờ gửi báo cáo tổng kết hàng ngày (Giờ Việt Nam UTC+7)
DAILY_REPORT_HOUR_VN = 6  # 06:00 sáng
