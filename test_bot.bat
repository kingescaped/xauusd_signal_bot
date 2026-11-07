@echo off
chcp 65001 >nul
echo ======================================================
echo    KIỂM TRA KẾT NỐI VÀ QUÉT TÍN HIỆU VÀNG + BTC (M15)
echo ======================================================
echo 1. Gửi tin nhắn kiểm tra kết nối:
python main.py --test
echo.
echo 2. Chạy thử nghiệm quét nến 1 lần (Vàng + BTC):
python main.py --once
echo.
echo 3. Gửi thử báo cáo tổng kết 24h:
python main.py --report
echo.
echo Kiểm tra hoàn tất!
pause
