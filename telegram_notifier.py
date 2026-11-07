import requests
import os
import config

def send_telegram_message(text):
    """
    Gửi tin nhắn định dạng Markdown vào nhóm Telegram.
    """
    if config.TELEGRAM_BOT_TOKEN == "DIEN_BOT_TOKEN_VAO_DAY" or config.TELEGRAM_GROUP_ID == "DIEN_GROUP_ID_VAO_DAY":
        print("[LỖI] Bạn chưa điền BOT_TOKEN hoặc GROUP_ID vào file config.py!")
        return False
        
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_GROUP_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        res_data = response.json()
        if res_data.get("ok"):
            print("[THÀNH CÔNG] Đã gửi tin nhắn vào nhóm Telegram.")
            return True
        else:
            print(f"[THẤT BẠI] Lỗi Telegram API: {res_data.get('description')}")
            return False
    except Exception as e:
        print(f"[LỖI] Kết nối Telegram thất bại: {e}")
        return False

def send_telegram_photo(photo_path, caption):
    """
    Gửi ảnh biểu đồ kèm theo ghi chú vào nhóm Telegram.
    """
    if not os.path.exists(photo_path):
        return send_telegram_message(caption)
        
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendPhoto"
    data = {
        "chat_id": config.TELEGRAM_GROUP_ID,
        "caption": caption,
        "parse_mode": "Markdown"
    }
    
    try:
        with open(photo_path, "rb") as photo_file:
            files = {"photo": photo_file}
            response = requests.post(url, data=data, files=files, timeout=20)
            res_data = response.json()
            if res_data.get("ok"):
                print("[THÀNH CÔNG] Đã gửi ảnh biểu đồ vào nhóm Telegram.")
                return True
            else:
                print(f"[THẤT BẠI] Lỗi gửi ảnh: {res_data.get('description')}")
                return False
    except Exception as e:
        print(f"[LỖI] Kết nối gửi ảnh Telegram thất bại: {e}")
        return False

def test_telegram():
    """Hàm gửi thử nghiệm để kiểm tra kết nối tới Nhóm."""
    test_msg = (
        "🤖 *[KRONOS BOT] - KẾT NỐI THÀNH CÔNG!*\n\n"
        "Chào mừng bạn! Bot tín hiệu nến Vàng đã sẵn sàng hoạt động trong nhóm này.\n"
        f"📊 *Sản phẩm theo dõi:* `{config.SYMBOL}`\n"
        f"⏰ *Khung thời gian:* `{config.TIMEFRAME}`"
    )
    return send_telegram_message(test_msg)

if __name__ == "__main__":
    test_telegram()
