# KRONOS GOLD & BITCOIN SIGNAL BOT (M15)

Hệ thống tự động theo dõi, phân tích kỹ thuật và bắn tín hiệu đa tài sản (**Vàng XAUUSD** và **Bitcoin BTCUSD**) trên khung nến **M15 (15 phút)**, kèm vẽ biểu đồ trực quan và **tổng hợp báo cáo hiệu suất TP/SL vào lúc 06:00 sáng hàng ngày**.

---

## 🌟 Tính năng nổi bật

1. **Đa tài sản (Multi-Assets):** Quét song song cả **Vàng (`GC=F`)** và **Bitcoin (`BTC-USD`)** trong cùng 1 lần chạy.
2. **Khung nến M15:** Bắt kịp các biến động ngắn hạn trong phiên.
3. **Biểu đồ nến trực quan:** Tự động vẽ nến kèm các mốc **Entry (Giá vào)**, **TP (Chốt lời - Xanh)** và **SL (Cắt lỗ - Đỏ)**.
4. **Theo dõi TP/SL tự động:** Lưu lại lịch sử tín hiệu vào `signals_history.json`. Mỗi cây nến mới sẽ kiểm tra xem các lệnh trước đó đã chạm TP hay dính SL chưa.
5. **Báo cáo tổng kết 06:00 Sáng hàng ngày:** Thống kê tổng số lệnh, số lệnh TP, số lệnh SL, tỷ lệ thắng (Win Rate %) trong 24 giờ qua và gửi về nhóm Telegram.
6. **Chạy hoàn toàn 0đ trên GitHub Actions:** Tự động lên lịch mỗi 15 phút.

---

## 🚀 Cách kích hoạt chạy thử trên GitHub Actions

1. Truy cập tab **Actions**: [github.com/kingescaped/xauusd_signal_bot/actions](https://github.com/kingescaped/xauusd_signal_bot/actions)
2. Chọn workflow **"Kronos Gold & Bitcoin Telegram Signal Bot (M15)"**.
3. Nhấn nút **"Run workflow"** ➔ Bot sẽ quét cả Vàng và Bitcoin, vẽ biểu đồ và gửi vào nhóm Telegram của bạn.
