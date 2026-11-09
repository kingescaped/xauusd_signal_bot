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
2. Chọn workflow **"Kronos-base Gold & Bitcoin Signal Bot"**.
3. Nhấn nút **"Run workflow"** ➔ Mô hình Kronos-base sẽ quét nến BTC (M15) & Vàng (1H) và gửi ảnh biểu đồ trực quan vào nhóm Telegram của bạn!
