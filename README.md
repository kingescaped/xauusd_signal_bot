# KRONOS-BASE GOLD & BITCOIN SIGNAL BOT (M15)

Dự án ứng dụng mô hình nền tảng AI tài chính chính thức **Kronos-base** (`shiyu-coder/Kronos`) cùng bộ mã hóa **`Kronos-Tokenizer-base`** từ Hugging Face để tự động đọc nến, dự phóng chuỗi K-line tương lai cho **Vàng (XAUUSD)** và **Bitcoin (BTCUSD)** trên khung nến **M15 (15 phút)**.

---

## 🧠 Cốt lõi AI: Kronos-base Foundation Model

* **Mô hình chính thức:** [`NeoQuasar/Kronos-base`](https://huggingface.co/NeoQuasar/Kronos-base) (~100 triệu tham số, huấn luyện trên 12 tỷ nến từ 45 sàn toàn cầu).
* **Bộ mã hóa K-line:** [`NeoQuasar/Kronos-Tokenizer-base`](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base).
* **Cơ chế suy luận:**
  1. Nạp chuỗi 200 nến OHLCV gần nhất vào Tokenizer.
  2. Mô hình Transformer tự hồi quy (Autoregressive) dự phóng trước **12 cây nến tiếp theo** (tương đương 3 giờ giao dịch trong khung M15).
  3. Chạy mô phỏng Monte Carlo để tính toán xác suất xu hướng, điểm chốt lời (TP) và cắt lỗ (SL) thích ứng theo râu nến dự phóng.

---

## 🌟 Tính năng hệ thống

1. **Đa tài sản:** Tự động phân tích cả **Vàng (`XAUUSD`)** và **Bitcoin (`BTCUSD`)**.
2. **Biểu đồ nến trực quan:** Tự động vẽ nến kèm các mốc **Entry**, **TP (Xanh)** và **SL (Đỏ)** gửi vào Telegram.
3. **Theo dõi TP/SL tự động:** Lưu lại lịch sử tín hiệu vào `signals_history.json`. Mỗi cây nến mới sẽ kiểm tra xem các lệnh trước đó đã chạm TP hay dính SL chưa.
4. **Báo cáo tổng kết 06:00 Sáng hàng ngày:** Thống kê tổng số lệnh, số lệnh TP, số lệnh SL, tỷ lệ thắng (Win Rate %) trong 24 giờ qua và gửi về nhóm Telegram.
5. **Chạy hoàn toàn 0đ trên GitHub Actions:** Có cơ chế cache trọng số mô hình giúp chạy cực nhanh (~20 giây mỗi lần).

---

## 🚀 Kích hoạt chạy thử trên GitHub Actions

1. Truy cập tab **Actions**: [github.com/kingescaped/xauusd_signal_bot/actions](https://github.com/kingescaped/xauusd_signal_bot/actions)
2. Chọn workflow **"Kronos-base Gold & Bitcoin Signal Bot (M15)"**.
3. Nhấn nút **"Run workflow"** ➔ Mô hình Kronos-base sẽ nạp, đọc nến và bắn tín hiệu kèm biểu đồ vào nhóm Telegram của bạn.
