@echo off
chcp 65001 >nul
echo ======================================================
echo       KIỂM TRA KẾT NỐI VÀO NHÓM TELEGRAM
echo ======================================================
python main.py --test
echo.
echo ======================================================
echo       KIỂM TRA QUÉT TÍN HIỆU 1 LẦN THỰC TẾ
echo ======================================================
python main.py --once
pause
