# KRONOS GOLD TELEGRAM SIGNAL BOT (XAUUSD)

Dự án tự động lấy dữ liệu nến Vàng thế giới (`GC=F` / `XAUUSD`) qua `yfinance`, đưa vào mô hình AI phân tích xu hướng, vẽ biểu đồ kỹ thuật và tự động bắn cảnh báo vào Nhóm Telegram.

---

## 1. Cách cấu hình bot nhắn vào NHÓM (Group) Telegram

### Bước 1: Thêm Bot vào nhóm của bạn
1. Mở nhóm Telegram mà bạn muốn nhận thông báo.
2. Thêm con bot bạn vừa tạo vào nhóm.
3. Cấp quyền **Admin (Quản trị viên)** cho bot để nó có quyền gửi tin nhắn và gửi ảnh.

### Bước 2: Lấy Group ID (ID nhóm)
> ⚠️ **LƯU Ý:** ID của Nhóm Telegram **LUÔN BẮT ĐẦU BẰNG DẤU TRỪ (`-`)**, thường có dạng: `-100xxxxxxxxxx`.

**Cách lấy ID nhóm nhanh nhất:**
* **Cách 1:** Thêm con bot `@RawDataBot` hoặc `@MissRose_bot` vào nhóm của bạn ➔ Nó sẽ gửi ngay 1 tin nhắn hiển thị `Chat ID: -100xxxxxxxxxx` ➔ Copy dãy số có cả dấu trừ đó ➔ Xong thì kích bot đó ra khỏi nhóm.
* **Cách 2:** Gửi một tin nhắn bất kỳ vào nhóm (ví dụ: `test`). Mở trình duyệt web gõ link sau (thay `<TOKEN>` bằng token bot của bạn):
  `https://api.telegram.org/bot<TOKEN>/getUpdates`
  Tìm trong kết quả chữ `"chat":{"id": -100xxxxxxxxxx`.

### Bước 3: Điền cấu hình vào file `config.py`
Mở file `config.py` và điền:
```python
TELEGRAM_BOT_TOKEN = "7123456789:AAFn..."
TELEGRAM_GROUP_ID = "-1001234567890"  # Phải có dấu trừ ở đầu
```

---

## 2. Hướng dẫn chạy Bot

1. **Cài đặt thư viện:** Nhấp đúp vào file `install_requirements.bat` (hoặc mở terminal gõ `pip install -r requirements.txt`).
2. **Kiểm tra kết nối:** Nhấp đúp vào file `test_bot.bat` để bot gửi 1 tin nhắn test vào nhóm và chạy thử quét nến 1 lần.
3. **Chạy chính thức 24/7:** Nhấp đúp vào file `run_bot.bat`. Bot sẽ tự động chạy ngầm và quét nến vào đầu mỗi tiếng.
