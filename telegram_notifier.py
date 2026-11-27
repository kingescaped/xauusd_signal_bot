import os
import time
import requests
import config

_PLACEHOLDERS = {"DIEN_BOT_TOKEN_VAO_DAY", "DIEN_GROUP_ID_VAO_DAY"}


def _has_valid_credentials():
    token = str(getattr(config, "TELEGRAM_BOT_TOKEN", "") or "").strip()
    group_id = str(getattr(config, "TELEGRAM_GROUP_ID", "") or "").strip()
    if not token or not group_id or token in _PLACEHOLDERS or group_id in _PLACEHOLDERS:
        print("[LỖI] Thiếu TELEGRAM_BOT_TOKEN hoặc TELEGRAM_GROUP_ID!")
        return False
    return True


def _post_with_retry(request_fn, label, retries=3, base_delay=1.5):
    for attempt in range(1, retries + 1):
        try:
            response = request_fn()
            res_data = response.json()
            if res_data.get("ok"):
                print(f"[THÀNH CÔNG] Đã {label} vào nhóm Telegram.")
                return True
            status = getattr(response, "status_code", 200)
            retryable = status == 429 or status >= 500
            if not retryable or attempt == retries:
                print(f"[THẤT BẠI] Lỗi Telegram API ({status}): {res_data.get('description')}")
                return False
            wait_sec = min(10.0, float((res_data.get("parameters") or {}).get("retry_after", base_delay * attempt)))
        except requests.RequestException as exc:
            if attempt == retries:
                print(f"[LỖI] Kết nối Telegram thất bại sau {retries} lần: {exc}")
                return False
            wait_sec = base_delay * attempt
        time.sleep(wait_sec)
    return False


def send_telegram_message(text):
    """Gửi tin nhắn định dạng HTML vào nhóm Telegram."""
    if not _has_valid_credentials():
        return False
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": config.TELEGRAM_GROUP_ID, "text": text, "parse_mode": "HTML"}
    return _post_with_retry(lambda: requests.post(url, json=payload, timeout=10), "gửi tin nhắn")


def send_telegram_photo(photo_path, caption):
    """Gửi ảnh biểu đồ kèm theo ghi chú HTML vào nhóm Telegram."""
    if not _has_valid_credentials():
        return False
    if not os.path.exists(photo_path):
        return send_telegram_message(caption)
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendPhoto"
    data = {"chat_id": config.TELEGRAM_GROUP_ID, "caption": caption, "parse_mode": "HTML"}

    def _send_file():
        with open(photo_path, "rb") as photo_file:
            return requests.post(url, data=data, files={"photo": photo_file}, timeout=20)

    return _post_with_retry(_send_file, "gửi ảnh biểu đồ")


def test_telegram():
    """Hàm gửi thử nghiệm để kiểm tra kết nối tới Nhóm."""
    test_msg = (
        "🤖 <b>[VÀNG Đánh BTC] - KẾT NỐI THÀNH CÔNG!</b>\n\n"
        "Chào mừng bạn! Hệ thống bot tín hiệu AI Kronos đã kết nối thành công với nhóm Telegram này.\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "📊 <b>Sản phẩm theo dõi:</b>\n"
        "🟡 <b>Vàng (XAUUSD):</b> Khung H1 (1 Giờ)\n"
        "🟠 <b>Bitcoin (BTCUSD):</b> Khung M15 (15 Phút)\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "⏰ <b>Lịch báo cáo:</b> 06:00 Sáng hàng ngày &amp; Thứ Bảy hàng tuần."
    )
    return send_telegram_message(test_msg)


if __name__ == "__main__":
    test_telegram()
