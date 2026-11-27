from datetime import timezone
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from market_time import VN_TZ, format_timeframe_label, to_vn_datetime


def _convert_plot_times_vn(time_col):
    if hasattr(time_col, "dt") and getattr(time_col.dt, "tz", None) is not None:
        return time_col.dt.tz_convert(VN_TZ)
    raw = [time_col.iloc[i] for i in range(len(time_col))]
    vn_vals = [to_vn_datetime(v, default_tz=timezone.utc) or v for v in raw]
    return pd.to_datetime([dt.isoformat() if hasattr(dt, "isoformat") else str(dt) for dt in vn_vals])


def create_signal_chart(df, symbol_name, symbol_label, timeframe, unit, predicted_direction, current_price, tp, sl, output_path="signal_chart.png", digits=2):
    try:
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)

        plot_df = df.tail(35).copy()
        plot_df["time"] = _convert_plot_times_vn(plot_df["time"])
        plot_df.set_index("time", inplace=True)
        plot_df.rename(columns={
            "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
        }, inplace=True)

        mc = mpf.make_marketcolors(up="#26a69a", down="#ef5350", edge="inherit", wick="inherit", volume="#424242")
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds", marketcolors=mc, gridstyle=":",
            gridcolor="#2d3748", facecolor="#1a202c", edgecolor="#2d3748",
        )
        hlines = dict(hlines=[tp, current_price, sl], colors=["#00e676", "#ffd600", "#ff1744"], linestyle="--", linewidths=1.2)
        tf_label = format_timeframe_label(timeframe)
        clean_dir = predicted_direction.replace("🟢", "").replace("🔴", "").replace("⚪", "").strip()
        title = (
            f"{symbol_label} ({tf_label} - Giờ VN) | KRONOS AI SIGNAL: {clean_dir}\n"
            f"Entry: {current_price:.{digits}f} | TP: {tp:.{digits}f} | SL: {sl:.{digits}f}"
        )

        mpf.plot(
            plot_df, type="candle", style=style, hlines=hlines,
            title=dict(title=title, color="white", size=10),
            ylabel=f"Price ({unit})",
            savefig=dict(fname=output_path, dpi=130, bbox_inches="tight", facecolor="#1a202c"),
        )
        return output_path
    except Exception as e:
        print(f"[CẢNH BÁO] Lỗi khi tạo biểu đồ cho {symbol_name}: {e}")
        return None
    finally:
        plt.close("all")
