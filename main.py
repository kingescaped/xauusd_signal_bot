import argparse
import html
import os
import sys
import time
from datetime import datetime, timezone
import schedule
import config
from bot_state import has_sent_report, mark_report_sent
from chart_visualizer import create_signal_chart
from fetch_data import get_candles, split_closed_and_live
from hf_repo_sync import sync_base_to_private_repo
from history_tracker import (
    add_new_signal,
    generate_daily_report,
    generate_weekly_report,
    has_open_signals,
    load_history,
    remove_signal,
    update_open_signals,
)
from kronos_predictor import predict_future
from market_time import VN_TZ, format_timeframe_label, is_market_open
from meta_model import evaluate_meta_signal, train_weekly_meta_model
from telegram_notifier import send_telegram_message, send_telegram_photo, test_telegram


def run_weekly_meta_training():
    state = train_weekly_meta_model(load_history())
    mode_txt = "ĐÃ BẬT LỌC CHẶN (<52%)" if state.get("filter_active") else "GHI NHẬN CHỜ ĐỦ >=100 LỆNH SẠCH"
    print(
        f"[META-MODEL TUẦN] Đã cập nhật bộ lọc: {state.get('clean_trade_count', 0)}/"
        f"{state.get('min_clean_trades', 100)} lệnh sạch -> Trạng thái: {mode_txt}"
    )
    return state


def check_and_send_daily_report_if_due(now_vn=None):
    now_vn = now_vn or datetime.now(VN_TZ)
    if now_vn.hour != config.DAILY_REPORT_HOUR_VN:
        return

    today_key = now_vn.strftime("%Y-%m-%d")
    if not has_sent_report("daily_report", today_key):
        print(f"[{now_vn.strftime('%H:%M:%S')}] Khung 06h sáng VN -> Đang tổng hợp báo cáo ngày hôm qua...")
        if send_telegram_message(generate_daily_report()):
            mark_report_sent("daily_report", today_key)
            print("Đã gửi báo cáo ngày hôm qua vào nhóm!")

    if now_vn.weekday() == 5 and not has_sent_report("weekly_report", today_key):
        print(f"[{now_vn.strftime('%H:%M:%S')}] Sáng Thứ Bảy -> Đang tổng hợp báo cáo tổng kết tuần...")
        try:
            run_weekly_meta_training()
        except Exception as exc:
            print(f"[CẢNH BÁO META-MODEL] Huấn luyện tuần gặp lỗi nhẹ: {exc}")
        if send_telegram_message(generate_weekly_report()):
            mark_report_sent("weekly_report", today_key)
            print("Đã gửi báo cáo tổng kết tuần vào nhóm!")


def _dispatch_telegram(rendered_chart_path, caption):
    if rendered_chart_path and os.path.exists(rendered_chart_path):
        return bool(send_telegram_photo(rendered_chart_path, caption))
    return bool(send_telegram_message(caption))


def _resolve_order_num_for_signal(signal_id):
    try:
        for sig in reversed(load_history()):
            if sig.get("id") == signal_id:
                return int(sig.get("order_num", 1))
    except Exception:
        pass
    return 1


def _build_signal_caption(item, result, tf_label, order_num=1):
    nhan_dinh = "Nhận định của Vàng" if item["name"] == "XAUUSD" else "Nhận định của BTC"
    digits = int(item.get("digits", 2))
    adx_info = f" | <b>ADX:</b> <code>{result.get('adx', 0):.1f}</code>" if "adx" in result else ""
    agree_info = f" | Đồng thuận: <code>{result['sample_agreement']}%</code>" if "sample_agreement" in result else ""
    meta_info = f"\n🧠 <b>Xác suất thắng Meta-AI:</b> <code>{result['meta_win_prob']}%</code>" if "meta_win_prob" in result else ""
    win_vn = result.get("forecast_window_vn") or ""
    horizon_txt = f"{item['predict_len']} nến tới: <code>{html.escape(win_vn)}</code>" if win_vn else f"{item['predict_len']} nến tới"
    dca_line = "\n🔄 <b>Vị thế:</b> <b>LỆNH #2 (DCA cùng xu hướng — lưu ý chia khối lượng)</b>" if order_num > 1 else ""
    unit = html.escape(str(item["unit"]))
    return (
        f"🎯 <b>[VÀNG Đánh BTC] - TÍN HIỆU VÀO LỆNH {html.escape(item['label'])}</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Sản phẩm:</b> <code>{html.escape(item['name'])}</code> | <b>Khung:</b> <code>{html.escape(tf_label)}</code>{dca_line}\n"
        f"💰 <b>Giá vào (Entry):</b> <code>{result['current_price']:.{digits}f} {unit}</code>\n"
        f"🚀 <b>Tín hiệu:</b> <b>{html.escape(result['direction'])}</b>\n"
        f"📈 <b>Độ tin cậy:</b> <code>{result['confidence']}%</code> (Đạt chuẩn ≥ {config.CONFIDENCE_THRESHOLD}%){agree_info}{meta_info}\n"
        f"🔮 <b>Kỳ vọng:</b> <code>{result['pct_change']:+.2f}%</code> ({horizon_txt})\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>Mục tiêu (TP):</b> <code>{result['tp']:.{digits}f} {unit}</code>\n"
        f"🛑 <b>Cắt lỗ (SL):</b> <code>{result['sl']:.{digits}f} {unit}</code>\n"
        f"⚡ <b>Biến động ATR:</b> <code>{result['atr']}</code>{adx_info}\n\n"
        f"💡 <b>{nhan_dinh}:</b> Tín hiệu theo đà xu hướng đang đẹp, kỳ vọng đạt mục tiêu TP!\n\n"
        "⚠️ <i>Hệ thống [VÀNG Đánh BTC] tự động theo dõi TP/SL và tổng kết lúc 06:00 sáng!</i>"
    )


def _build_force_caption(item, result, tf_label, block_note=None):
    nhan_dinh = "Nhận định của Vàng" if item["name"] == "XAUUSD" else "Nhận định của BTC"
    digits = int(item.get("digits", 2))
    unit = html.escape(str(item["unit"]))
    ai_note = "" if result.get("ai_predicted", False) else "\n⚠️ <b>[CẢNH BÁO: AI LVQ Std BOT không khả dụng, không tạo tín hiệu giao dịch]</b>"
    if block_note:
        summary_text = f"Kèo phân tích đang thuận xu hướng nhưng <b>không mở lệnh mới</b> ({html.escape(block_note)})."
    else:
        summary_text = "Thị trường đang Sideway / chưa đạt chuẩn xác suất thắng cao. Tiếp tục quan sát!"
    return (
        f"📊 <b>[VÀNG Đánh BTC] - CẬP NHẬT PHÂN TÍCH {html.escape(item['label'])}</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Sản phẩm:</b> <code>{html.escape(item['name'])}</code> | <b>Khung:</b> <code>{html.escape(tf_label)}</code>\n"
        f"💰 <b>Giá hiện tại:</b> <code>{result['current_price']:.{digits}f} {unit}</code>\n"
        f"🎯 <b>Trạng thái:</b> <b>{html.escape(result['direction'])}</b>\n"
        f"📈 <b>Độ tin cậy:</b> <code>{result['confidence']}%</code> <i>(Cần ≥ {config.CONFIDENCE_THRESHOLD}% để vào lệnh)</i>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 <b>{nhan_dinh}:</b> {summary_text}{ai_note}"
    )


def _render_symbol_chart(item, df, result):
    symbol_name = item["name"]
    chart_path = os.path.join(os.path.dirname(__file__), f"chart_{symbol_name.lower()}.png")
    return create_signal_chart(
        df=df, symbol_name=symbol_name, symbol_label=item["label"], timeframe=item["timeframe"],
        unit=item["unit"], predicted_direction=result["direction"], current_price=result["current_price"],
        tp=result["tp"], sl=result["sl"], output_path=chart_path, digits=int(item.get("digits", 2)),
    )


def _check_meta_gate(symbol_name, result):
    try:
        hist = load_history()
        next_order = sum(1 for s in hist if s.get("symbol") == symbol_name and s.get("status") == "OPEN") + 1
    except Exception:
        hist, next_order = None, 1
    decision = evaluate_meta_signal(symbol_name, result, history=hist, order_num=next_order)
    mode_label = "LỌC CHẶN" if decision.filter_active else "GHI NHẬN"
    print(
        f"-> {symbol_name}: [META-MODEL {mode_label}] Win Prob = {decision.win_prob * 100:.1f}% "
        f"({decision.clean_trades}/{getattr(config, 'META_MIN_CLEAN_TRADES', 100)} lệnh sạch)"
    )
    return decision


def _persist_and_dispatch_signal(item, df, time_str, result, manual_force):
    symbol_name, timeframe = item["name"], item["timeframe"]
    tf_label = format_timeframe_label(timeframe)
    added, reason = add_new_signal(
        symbol=symbol_name, direction=result["direction"], entry_price=result["current_price"],
        tp=result["tp"], sl=result["sl"], time_str=time_str, timeframe=timeframe,
        atr=result.get("atr", 1.0), metadata=result,
    )
    if added:
        order_num = _resolve_order_num_for_signal(reason)
        rendered = _render_symbol_chart(item, df, result)
        if _dispatch_telegram(rendered, _build_signal_caption(item, result, tf_label, order_num=order_num)):
            print(f"-> {symbol_name}: 🚀 Bắn tín hiệu #{order_num} vào nhóm Telegram thành công! (Mã: {reason})")
        else:
            remove_signal(reason)
            print(f"-> {symbol_name}: [LỖI] Gửi Telegram thất bại -> Đã hoàn tác không lưu tín hiệu {reason}.")
        return
    print(f"-> {symbol_name}: ⏸️ {reason}")
    if manual_force:
        rendered = _render_symbol_chart(item, df, result)
        _dispatch_telegram(rendered, _build_force_caption(item, result, tf_label, block_note=reason))


def _handle_symbol_result(item, df, time_str, result, manual_force=False):
    symbol_name, timeframe = item["name"], item["timeframe"]
    allow_persist = not bool(result.get("_off_schedule_h1", False))
    ai_ok = bool(result.get("ai_predicted", False))
    is_high_conf = ai_ok and result["confidence"] >= config.CONFIDENCE_THRESHOLD and "SIDEWAY" not in result["direction"]
    if not is_high_conf and not manual_force:
        msg = "AI LVQ Std BOT không khả dụng -> bỏ qua tín hiệu." if not ai_ok else f"Độ tin cậy {result['confidence']}% (< {config.CONFIDENCE_THRESHOLD}%) hoặc Sideway -> Giữ yên lặng tuyệt đối."
        print(f"-> {symbol_name}: {msg}")
        return

    tf_label = format_timeframe_label(timeframe)
    if is_high_conf and allow_persist:
        meta_decision = _check_meta_gate(symbol_name, result)
        if meta_decision.blocked:
            print(f"-> {symbol_name}: 🛡️ {meta_decision.reason}")
            if manual_force:
                rendered = _render_symbol_chart(item, df, result)
                _dispatch_telegram(rendered, _build_force_caption(item, result, tf_label, block_note=meta_decision.reason))
            return
        _persist_and_dispatch_signal(item, df, time_str, result, manual_force)
        return

    note = "Đang kiểm tra ngoài khung đóng nến H1" if (is_high_conf and not allow_persist) else None
    print(f"-> {symbol_name}: [Có cờ --force] Gửi ảnh phân tích.")
    rendered = _render_symbol_chart(item, df, result)
    _dispatch_telegram(rendered, _build_force_caption(item, result, tf_label, block_note=note))


def _refresh_open_positions_fine(symbol_name, ticker, timeframe, now_utc):
    if has_open_signals(symbol_name, recent_minutes=120, now_utc=now_utc):
        df_1m = get_candles(ticker, timeframe="1m", period="1d", limit=180)
        if df_1m is not None and len(df_1m) > 0:
            last_1m = df_1m.iloc[-1]
            update_open_signals(symbol_name, float(last_1m["high"]), float(last_1m["low"]), float(last_1m["close"]), str(last_1m["time"]), df=df_1m)
            return
    if timeframe.lower() == "1h" and has_open_signals(symbol_name):
        df_15m = get_candles(ticker, timeframe="15m", period="2d", limit=24)
        if df_15m is not None and len(df_15m) > 0:
            last_15m = df_15m.iloc[-1]
            update_open_signals(symbol_name, float(last_15m["high"]), float(last_15m["low"]), float(last_15m["close"]), str(last_15m["time"]), df=df_15m)


def _process_symbol(item, now_vn, manual_force):
    symbol_name, ticker, label = item["name"], item["ticker"], item["label"]
    timeframe, pred_len, period = item["timeframe"], item["predict_len"], item["period"]
    now_utc = now_vn.astimezone(timezone.utc)

    market_open, market_msg = is_market_open(symbol_name, now_vn)
    if not market_open:
        print(f"[{symbol_name}] ⏸️ {market_msg} -> Tạm dừng quét & giữ yên lặng tuyệt đối.")
        return

    is_mid_hour_h1 = timeframe.lower() == "1h" and now_vn.minute >= 15
    if is_mid_hour_h1 and not manual_force:
        print(f"[{symbol_name} - 1H] Chưa đóng nến 1H (phút {now_vn.minute}) -> Chờ đến đầu tiếng tiếp theo.")
        _refresh_open_positions_fine(symbol_name, ticker, "15m", now_utc)
        df_check = get_candles(ticker, timeframe="15m", period="2d", limit=24)
        if df_check is not None and len(df_check) > 0:
            last_c = df_check.iloc[-1]
            update_open_signals(symbol_name, float(last_c["high"]), float(last_c["low"]), float(last_c["close"]), str(last_c["time"]), df=df_check)
        return

    print(f"\n--- Đang xử lý: {label} (Khung {timeframe.upper()}) ---")
    _refresh_open_positions_fine(symbol_name, ticker, timeframe, now_utc)
    fetch_limit = getattr(config, "INDICATOR_CANDLE_LIMIT", config.CANDLE_LIMIT) + 1
    df = get_candles(ticker, timeframe=timeframe, period=period, limit=fetch_limit)
    if df is None or len(df) == 0:
        print(f"[BỎ QUA] Không đủ nến cho {label}.")
        return

    latest = df.iloc[-1]
    update_open_signals(symbol_name, float(latest["high"]), float(latest["low"]), float(latest["close"]), str(latest["time"]), df=df)

    closed_df, live_price, _, had_forming = split_closed_and_live(df, timeframe=timeframe, now_utc=now_utc)
    if closed_df is None or len(closed_df) < 25:
        print(f"[BỎ QUA] Không đủ nến đã đóng cho {label}.")
        return
    if had_forming:
        print(f"-> [{symbol_name}] Đã tách nến đang chạy, AI đọc {len(closed_df)} nến đã đóng (Entry live: {live_price}).")

    closed_time_str = str(closed_df.iloc[-1]["time"])
    result = dict(predict_future(closed_df, symbol_name, timeframe=timeframe, pred_len=pred_len, live_price=live_price))
    if is_mid_hour_h1:
        result["_off_schedule_h1"] = True
    print(f"-> Phân tích {symbol_name} ({timeframe}): {result['direction']} | Độ tin cậy: {result['confidence']}%")
    _handle_symbol_result(item, closed_df, closed_time_str, result, manual_force)


def run_single_cycle(manual_force=False, now_vn=None):
    now_vn = now_vn or datetime.now(VN_TZ)
    print(f"\n[{now_vn.strftime('%Y-%m-%d %H:%M:%S')} VN] Bắt đầu chu kỳ quét nến...")
    if manual_force:
        print("⚡ [CHẾ ĐỘ ÉP BUỘC] Luôn gửi ảnh và phân tích kể cả khi Sideway.")
    else:
        print(f"🤖 [CHẠY TỰ ĐỘNG] Giữ im lặng khi Sideway/cuối tuần. Chỉ gửi tin nhắn khi có kèo đẹp độ tin cậy >= {config.CONFIDENCE_THRESHOLD}%.")

    for item in config.SYMBOLS:
        try:
            _process_symbol(item, now_vn, manual_force)
        except Exception as exc:
            print(f"[LỖI CHU KỲ] Cặp {item.get('name')} gặp lỗi ngoài dự kiến: {exc}")

    try:
        check_and_send_daily_report_if_due(now_vn=now_vn)
    except Exception as exc:
        print(f"[LỖI BÁO CÁO] Không thể kiểm tra/gửi báo cáo định kỳ: {exc}")


def _build_cli_parser():
    parser = argparse.ArgumentParser(description="VÀNG Đánh BTC Signal Bot")
    parser.add_argument("--test", action="store_true", help="Kiểm tra kết nối Telegram vào nhóm")
    parser.add_argument("--report", action="store_true", help="Gửi ngay báo cáo hiệu suất ngày hôm qua vào nhóm")
    parser.add_argument("--report-week", action="store_true", help="Gửi ngay báo cáo tổng kết tuần (Thứ 2 - Thứ 6) vào nhóm")
    parser.add_argument("--train-meta", action="store_true", help="Huấn luyện lại Meta-Model lọc kèo từ lịch sử lệnh sạch")
    parser.add_argument("--sync-hf", action="store_true", help="Đồng bộ Kronos-base sang repo Private Hugging Face và tạo nhánh")
    parser.add_argument("--once", action="store_true", help="Chạy quét 1 chu kỳ ngay bây giờ")
    parser.add_argument("--force", action="store_true", help="Ép buộc gửi ảnh biểu đồ kể cả lúc Sideway (cho chế độ bấm tay)")
    return parser


def main():
    args = _build_cli_parser().parse_args()
    if args.test:
        if not test_telegram():
            sys.exit(1)
        return
    if args.report:
        if not send_telegram_message(generate_daily_report()):
            sys.exit(1)
        return
    if args.report_week:
        if not send_telegram_message(generate_weekly_report()):
            sys.exit(1)
        return
    if args.train_meta:
        run_weekly_meta_training()
        return
    if args.sync_hf:
        sync_base_to_private_repo()
        return
    if args.once:
        run_single_cycle(manual_force=args.force)
        return

    print("==================================================")
    print("   [VÀNG Đánh BTC] BOT: BTC (M15) & VÀNG (1H)     ")
    print("   - BTCUSD: Khung M15 (quét mỗi 15 phút)         ")
    print("   - XAUUSD: Khung H1  (quét đầu mỗi tiếng)       ")
    print("   - Báo cáo định kỳ: 06:00 Sáng hàng ngày        ")
    print("==================================================")
    run_single_cycle(manual_force=False)
    schedule.every(15).minutes.do(lambda: run_single_cycle(manual_force=False))
    while True:
        try:
            schedule.run_pending()
        except Exception as exc:
            print(f"[LỖI SCHEDULER] Chu kỳ chạy nền gặp sự cố: {exc}")
        time.sleep(15)


if __name__ == "__main__":
    main()
