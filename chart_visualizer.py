import matplotlib
matplotlib.use('Agg')  # Chạy headless trên Linux / GitHub Actions
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import os

def create_signal_chart(df, predicted_direction, current_price, tp, sl, output_path="signal_chart.png"):
    """
    Vẽ biểu đồ nến 35 cây gần nhất kèm các đường Target (TP) và Cắt lỗ (SL),
    sau đó xuất ra file ảnh để gửi vào nhóm Telegram.
    """
    try:
        # Lấy 35 cây nến gần nhất
        plot_df = df.tail(35).copy()
        
        # Đảm bảo index là DatetimeIndex chuẩn cho mplfinance
        plot_df['time'] = pd.to_datetime(plot_df['time'])
        plot_df.set_index('time', inplace=True)
        
        plot_df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)

        # Style nến hiện đại (nền tối, nến xanh tăng / đỏ giảm)
        mc = mpf.make_marketcolors(
            up='#26a69a', down='#ef5350',
            edge='inherit',
            wick='inherit',
            volume='#424242'
        )
        s = mpf.make_mpf_style(
            base_mpf_style='nightclouds',
            marketcolors=mc,
            gridstyle=':',
            gridcolor='#2d3748',
            facecolor='#1a202c',
            edgecolor='#2d3748'
        )

        # Vẽ thêm các đường TP và SL lên biểu đồ
        hlines = dict(
            hlines=[tp, current_price, sl],
            colors=['#00e676', '#ffd600', '#ff1744'],
            linestyle='--',
            linewidths=1.2
        )

        title = f"XAUUSD (GOLD 1H) | SIGNAL: {predicted_direction}\nCurrent: ${current_price:.2f} | TP: ${tp:.2f} | SL: ${sl:.2f}"

        # Xuất file ảnh độ phân giải cao
        mpf.plot(
            plot_df,
            type='candle',
            style=s,
            hlines=hlines,
            title=dict(title=title, color='white', size=11),
            ylabel='Price ($/oz)',
            savefig=dict(fname=output_path, dpi=140, bbox_inches='tight', facecolor='#1a202c')
        )
        return output_path
    except Exception as e:
        print(f"[CẢNH BÁO] Lỗi khi tạo biểu đồ: {e}")
        return None
