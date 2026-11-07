import matplotlib
matplotlib.use('Agg')  # Chạy headless trên Linux / GitHub Actions
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import os

def create_signal_chart(df, symbol_name, symbol_label, timeframe, unit, predicted_direction, current_price, tp, sl, output_path="signal_chart.png"):
    """
    Vẽ biểu đồ nến 35 cây gần nhất cho Vàng (1H) hoặc Bitcoin (M15) kèm đường TP, SL.
    """
    try:
        plot_df = df.tail(35).copy()
        plot_df['time'] = pd.to_datetime(plot_df['time'])
        plot_df.set_index('time', inplace=True)
        
        plot_df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)

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

        hlines = dict(
            hlines=[tp, current_price, sl],
            colors=['#00e676', '#ffd600', '#ff1744'],
            linestyle='--',
            linewidths=1.2
        )

        tf_label = "M15" if timeframe.lower() == "15m" else "H1"
        title = f"{symbol_label} ({tf_label}) | KRONOS-BASE SIGNAL: {predicted_direction}\nCurrent: {current_price:.2f} | TP: {tp:.2f} | SL: {sl:.2f}"

        mpf.plot(
            plot_df,
            type='candle',
            style=s,
            hlines=hlines,
            title=dict(title=title, color='white', size=10),
            ylabel=f'Price ({unit})',
            savefig=dict(fname=output_path, dpi=130, bbox_inches='tight', facecolor='#1a202c')
        )
        return output_path
    except Exception as e:
        print(f"[CẢNH BÁO] Lỗi khi tạo biểu đồ cho {symbol_name}: {e}")
        return None
