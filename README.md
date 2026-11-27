# KRONOS-BASE BOT: BTC (M15) & VÀNG (1H)

Hệ thống giao dịch định lượng ứng dụng mô hình nền tảng AI tài chính **Kronos** kết hợp kiến trúc **3 tầng tự cải thiện**:
* 🟠 **Bitcoin (`BTCUSD`):** Chạy khung **M15 (15 phút)** với nhánh trọng số chuyên biệt `btc-m15`.
* 🟡 **Vàng (`XAUUSD`):** Chạy khung **H1 (1 Giờ)** với nhánh trọng số chuyên biệt `xau-h1`.

---

## 🧠 Kiến trúc AI 3 Bước (Self-Improving Pipeline)

### Bước 1 — Fine-tune 2 nhánh chuyên biệt hàng tháng trên Google Colab
* **Tổng quan:** Notebook [`colab/finetune_kronos_monthly.ipynb`](colab/finetune_kronos_monthly.ipynb) tự động kéo **1 năm nến M15 BTC** từ Binance Public API (`data-api.binance.vision`) và **2 năm (`730d`) nến H1 Vàng** từ Yahoo Finance (`GC=F`), tinh chỉnh riêng 2 bộ trọng số và đẩy lên repo Private [`kingescaped/lvq-std-BOT`](https://huggingface.co/kingescaped/lvq-std-BOT) (`btc-m15` và `xau-h1`). Chỉ ghi đè lên nhánh khi `val_loss` mới tốt hơn hoặc bằng bản hiện tại.
* **Điều kiện tiên quyết & kiểm tra đầu vào:**
  * Tài khoản Google Colab chọn **T4 GPU** (`Runtime → Change runtime type → T4 GPU`).
  * Khóa `HF_TOKEN` quyền **Write** được khai báo trong **Colab Secrets** (biểu tượng 🔑 bên trái Colab, bật *Notebook access*).
  * Tối thiểu `>= 426` cây nến đã đóng hợp lệ cho BTC M15 và `>= 414` cây nến đã đóng cho Vàng H1 (hệ thống tự động kiểm tra và loại bỏ nến đang chạy / NaN).
* **Cách tạo nếu thiếu:** Nếu chưa có `HF_TOKEN` quyền Write, tạo mới tại [Hugging Face Access Tokens](https://huggingface.co/settings/tokens) (chọn *Write*), sau đó dán vào cả **Colab Secrets** và [GitHub Actions Secrets](#điều-kiện-tiên-quyết--kiểm-tra-đầu-vào). Nếu nhánh `btc-m15` / `xau-h1` chưa có `model.safetensors`, ô số 2 của Notebook hoặc chế độ `sync-hf` trên GitHub Actions sẽ tự sao chép từ [`NeoQuasar/Kronos-base`](https://huggingface.co/NeoQuasar/Kronos-base).

### Bước 2 & Bước 3 — Meta-Model hàng tuần & Bộ lọc chặn kèo tự động
* **Tổng quan:** Mô-đun `meta_model.py` học xác suất thắng thực tế (`HIT_TP` vs `HIT_SL`) từ lịch sử tín hiệu sạch sau kiểm toán (`v3.3*`, `v4.*`) và tự động cập nhật vào mỗi sáng **Thứ Bảy 06:00 VN** (lưu vào `meta_model_state.json`):
  * **Dưới `< 100` lệnh sạch đã đóng:** Chỉ chấm điểm xác suất thắng (`meta_win_prob`) và ghi nhận vào `signals_history.json` (**chưa chặn kèo**).
  * **Đủ `>= 100` lệnh sạch đã đóng** (có đủ cả 2 nhãn TP và SL): Tự động bật bộ lọc, **chặn các kèo có xác suất thắng dự phóng `< 52%`**.
* **Điều kiện tiên quyết & kiểm tra đầu vào:** Bản ghi trong `signals_history.json` phải có trạng thái `HIT_TP` hoặc `HIT_SL`, có `strategy_version` thuộc thế hệ sạch (`v3.3*` / `v4.*`) và đầy đủ các chỉ số `confidence`, `pct_change`, `atr > 0`, `adx`, `sample_agreement`, `sample_std_pct`. Các bản ghi cũ thiếu metadata sẽ tự động được bỏ qua không gây lỗi.
* **Cách tạo nếu thiếu:** File `meta_model_state.json` được tạo tự động vào sáng Thứ Bảy hàng tuần hoặc tạo ngay bất kỳ lúc nào bằng cách chạy chế độ **`train-meta`** trong bảng hướng dẫn bên dưới.

### Cơ chế Chuyển đổi An toàn Không Gián Đoạn (Zero-Downtime Fallback)
* Cả 2 cặp dùng chung 1 bộ mã hóa từ nhánh Private [`kingescaped/lvq-std-BOT@tokenizer`](https://huggingface.co/kingescaped/lvq-std-BOT/tree/tokenizer) (tự động lùi về [`NeoQuasar/Kronos-Tokenizer-base`](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) nếu cần) nạp 1 lần duy nhất trong bộ nhớ.
* Khi quét từng cặp, bot ưu tiên nạp `kingescaped/lvq-std-BOT@btc-m15` (hoặc `xau-h1`) → nếu nhánh chưa sẵn sàng sẽ tự lùi về `kingescaped/lvq-std-BOT@main` → nếu thiếu quyền `HF_TOKEN` hoặc chưa đồng bộ `model.safetensors` sẽ **tự động lùi về `NeoQuasar/Kronos-base` công khai mà không phát sinh lỗi dừng bot**.

---

## 🚀 Kích hoạt chạy trên GitHub Actions

1. Truy cập tab **Actions**: [github.com/kingescaped/xauusd_signal_bot/actions](https://github.com/kingescaped/xauusd_signal_bot/actions)
2. Chọn workflow **"VÀNG Đánh BTC Signal Bot"**.
3. Nhấn nút **"Run workflow"**, chọn chế độ trong mục **`Chọn chế độ chạy:`** rồi bấm chạy.

---

## ⚙️ Giải thích các chế độ chạy (`Chọn chế độ chạy`)

| Chế độ | Lệnh thực thi | Mục đích & Hành vi chi tiết | Lưu trạng thái vào Repo |
| :--- | :--- | :--- | :---: |
| **`once`** *(mặc định)* | `python main.py --once` | **Quét nến tự động chuẩn:** Kiểm tra gửi báo cáo `06:00 VN` (kèm huấn luyện Meta-Model sáng Thứ Bảy), cập nhật TP/SL lệnh mở, quét AI bằng nến đã đóng (`BTC` M15 mọi chu kỳ; `XAU` H1 đầu giờ `phút < 15`), chấm điểm/lọc Meta-Model và gửi Telegram khi đạt chuẩn (`>= 75%`). | Có (`signals_history.json`, `bot_state.json`, `meta_model_state.json`) |
| **`sync-hf`** | `python main.py --sync-hf` | **Khởi tạo / đồng bộ Hugging Face Private:** Tự động kiểm tra repo Private `kingescaped/lvq-std-BOT`, sao chép `Kronos-base` (409 MB) lên `main`, `btc-m15`, `xau-h1` và sao chép `Kronos-Tokenizer-base` (15.1 MB) lên nhánh `tokenizer`. | Không (lưu trên Hugging Face) |
| **`train-meta`** | `python main.py --train-meta` | **Huấn luyện lại Meta-Model ngay:** Quét toàn bộ lệnh sạch đã đóng trong `signals_history.json`, cập nhật trọng số Logistic L2 và trạng thái kích hoạt (`>= 100` lệnh sạch) vào `meta_model_state.json`. | Có (`meta_model_state.json`) |
| **`report`** | `python main.py --report` | **Gửi báo cáo ngày ngay lập tức:** Tổng kết tín hiệu của **ngày hôm qua** (`00:00 – 23:59` giờ VN) và gửi vào nhóm Telegram. | Không |
| **`report-week`** | `python main.py --report-week` | **Gửi báo cáo tổng kết tuần ngay lập tức:** Thống kê kết quả Thứ Hai – Thứ Sáu kèm Win Rate từng mã gửi vào Telegram. | Không |
| **`force`** | `python main.py --once --force` | **Ép buộc gửi ảnh phân tích:** Bỏ qua giới hạn đầu giờ của Vàng H1 và luôn gửi biểu đồ + nhận định kể cả khi Sideway hoặc bị Meta-Model chặn *(vẫn nghỉ khi thị trường Vàng đóng cửa)*. | Chỉ lưu nếu mở kèo chuẩn hoặc chạm TP/SL |
| **`test`** | `python main.py --test` | **Kiểm tra kết nối Telegram:** Gửi tin nhắn thử nghiệm vào nhóm xác nhận Token & Group ID. | Không |

### Điều kiện tiên quyết & kiểm tra đầu vào

- **Giá trị đầu vào `action`:** Chọn một trong `once`, `sync-hf`, `train-meta`, `report`, `report-week`, `force`, `test` (mặc định `once`).
- **GitHub Secrets bắt buộc:** Thiết lập tại [Settings → Secrets and variables → Actions](https://github.com/kingescaped/xauusd_signal_bot/settings/secrets/actions):
  - `TELEGRAM_BOT_TOKEN`: Token bot tạo từ [@BotFather](https://t.me/BotFather) trên Telegram.
  - `TELEGRAM_GROUP_ID`: Chat ID nhóm Telegram nhận tín hiệu (dạng `-100...`).
  - `HF_TOKEN`: Token tạo tại [Hugging Face Tokens](https://huggingface.co/settings/tokens) — **cần quyền `Write`** nếu dùng chế độ `sync-hf` và Colab Fine-tune trên repo Private `kingescaped/lvq-std-BOT` (nếu tạm thời chưa điền, chế độ `once` vẫn tự động chạy bình thường bằng bản dự phòng `NeoQuasar/Kronos-base`).
