# KRONOS-BASE BOT: BTC (M15) & VÀNG (1H)

Dự án ứng dụng mô hình nền tảng AI tài chính chính thức **Kronos-base** (`shiyu-coder/Kronos`) cùng bộ mã hóa **`Kronos-Tokenizer-base`** để tự động đọc nến, phân tích đa khung thời gian:
* 🟠 **Bitcoin (BTCUSD):** Chạy khung **M15 (15 phút)** để lướt sóng nhanh.
* 🟡 **Vàng (XAUUSD):** Chạy khung **H1 (1 Giờ)** để bắt xu hướng chuẩn xác, tránh nhiễu.

---

## 🧠 Cốt lõi AI: Kronos-base Foundation Model

* **Mô hình chính thức:** [`NeoQuasar/Kronos-base`](https://huggingface.co/NeoQuasar/Kronos-base) (~100 triệu tham số).
* **Bộ mã hóa K-line:** [`NeoQuasar/Kronos-Tokenizer-base`](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base).
* **Cơ chế suy luận:**
  * **BTC (M15):** Dự phóng trước **12 cây nến M15** (3 giờ giao dịch tiếp theo).
  * **Vàng (1H):** Dự phóng trước **6 cây nến 1H** (6 giờ giao dịch tiếp theo).
  * Chạy mô phỏng Monte Carlo để tìm ra đường đi tối ưu của giá, tự động tính mốc **TP (Chốt lời - Xanh)** và **SL (Cắt lỗ - Đỏ)**.

---

## 🌟 Tính năng hệ thống

1. **Đa khung thời gian thông minh:** 
   * BTC được quét liên tục mỗi 15 phút.
   * Vàng được phân tích vào đầu mỗi tiếng (ngay sau khi nến 1H đóng cửa).
2. **Biểu đồ nến trực quan:** Tự động vẽ nến kèm các mốc **Entry**, **TP (Xanh)** và **SL (Đỏ)** gửi vào Telegram.
3. **Theo dõi TP/SL tự động:** Lưu lại lịch sử tín hiệu vào `signals_history.json`. Mỗi cây nến mới sẽ kiểm tra xem các lệnh trước đó đã chạm TP hay dính SL chưa.
4. **Báo cáo hiệu suất ngày (06:00 Sáng hàng ngày):** Tổng kết toàn bộ tín hiệu của ngày hôm trước (từ 00:00 đến 23:59), thống kê số lệnh TP/SL và tỷ lệ thắng Win Rate chuẩn xác.
5. **Báo cáo tổng kết tuần (06:00 Sáng Thứ Bảy):** Tổng hợp hiệu suất giao dịch cả tuần (từ Thứ Hai đến Thứ Sáu), thống kê tỷ lệ thắng tổng thể và chi tiết riêng cho Vàng & Bitcoin.
6. **Chạy hoàn toàn 0đ trên GitHub Actions:** Có cơ chế cache trọng số mô hình giúp chạy cực nhanh (~20 giây mỗi lần).

---

## 🚀 Kích hoạt chạy thử trên GitHub Actions

1. Truy cập tab **Actions**: [github.com/kingescaped/xauusd_signal_bot/actions](https://github.com/kingescaped/xauusd_signal_bot/actions)
2. Chọn workflow **"VÀNG Đánh BTC Signal Bot"**.
3. Nhấn nút **"Run workflow"**, chọn chế độ trong mục **`Chọn chế độ chạy:`** rồi bấm chạy.

---

## ⚙️ Giải thích các chế độ chạy (`Chọn chế độ chạy`)

| Chế độ | Lệnh thực thi | Mục đích & Hành vi chi tiết | Lưu lịch sử (`signals_history.json`) |
| :--- | :--- | :--- | :---: |
| **`once`** *(mặc định)* | `python main.py --once` | **Quét nến tự động chuẩn:** Kiểm tra gửi báo cáo `06:00–06:14 VN`, cập nhật TP/SL cho lệnh đang mở, rồi quét AI bằng nến đã đóng (`BTC` M15 mọi chu kỳ; `XAU` H1 vào đầu giờ `phút < 15`). Chỉ gửi Telegram khi có kèo đạt chuẩn (`>= 75%`, qua bộ lọc EMA/ADX/trượt giá); giữ im lặng khi Sideway hoặc ngoài giờ. | Có (nếu có kèo mới hoặc chạm TP/SL) |
| **`report`** | `python main.py --report` | **Gửi báo cáo ngày ngay lập tức:** Tổng kết toàn bộ tín hiệu của **ngày hôm qua** (`00:00 – 23:59` giờ VN) và gửi vào nhóm Telegram. Không quét nến mới. | Không |
| **`report-week`** | `python main.py --report-week` | **Gửi báo cáo tổng kết tuần ngay lập tức:** Thống kê kết quả Thứ Hai – Thứ Sáu (cuối tuần tự tính tuần vừa qua; ngày thường tính tuần hiện tại) kèm Win Rate từng mã và gửi vào Telegram. Không quét nến mới. | Không |
| **`force`** | `python main.py --once --force` | **Ép buộc gửi ảnh phân tích:** Bỏ qua giới hạn đầu giờ của Vàng H1 và luôn gửi ảnh biểu đồ + nhận định vào Telegram kể cả khi thị trường Sideway, độ tin cậy `< 75%`, hoặc AI tạm lỗi *(vẫn bỏ qua Vàng nếu đang đóng cửa thị trường)*. | Chỉ lưu nếu thực sự đạt kèo chuẩn hoặc chạm TP/SL |
| **`test`** | `python main.py --test` | **Kiểm tra kết nối Telegram:** Gửi một tin nhắn chào thử nghiệm vào nhóm để xác nhận Bot Token & Group ID hoạt động. Không tải model AI, không quét nến. | Không |

### 🔑 Điều kiện tiên quyết & kiểm tra đầu vào

- **Giá trị đầu vào `action`:** Bắt buộc chọn một trong `once`, `report`, `report-week`, `force`, `test` (mặc định `once`).
- **GitHub Secrets bắt buộc:** Thiết lập tại [Settings → Secrets and variables → Actions](https://github.com/kingescaped/xauusd_signal_bot/settings/secrets/actions) trước khi chạy bất kỳ chế độ nào:
  - `TELEGRAM_BOT_TOKEN`: Token bot lấy từ [@BotFather](https://t.me/BotFather) trên Telegram (bắt buộc — thiếu hoặc sai sẽ lỗi gửi tin).
  - `TELEGRAM_GROUP_ID`: Chat ID nhóm/kênh Telegram nhận thông báo, thường dạng `-100...` (thêm bot vào nhóm và lấy ID qua bot hỗ trợ hoặc Telegram `getUpdates`).
  - `HF_TOKEN` *(khuyến nghị)*: Token quyền Read tạo tại [Hugging Face Tokens](https://huggingface.co/settings/tokens) để tránh giới hạn tốc độ tải trọng số `Kronos-base`.
- **Dữ liệu & giờ giao dịch:**
  - `report` và `report-week` đọc từ `signals_history.json`; nếu file chưa tồn tại thì báo cáo hiển thị `0` lệnh (file được tạo/cập nhật tự động khi `once`/`force` phát sinh hoặc đóng lệnh).
  - `force` vẫn không gửi biểu đồ cho Vàng (`XAUUSD`) trong thời gian đóng cửa cuối tuần (`05:00` Thứ Bảy – `06:00` Thứ Hai VN) và giờ nghỉ kỹ thuật (`05:00–06:00` Thứ Ba – Thứ Sáu VN).
