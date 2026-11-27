from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover
    ZoneInfo = None
    ZoneInfoNotFoundError = Exception

VN_TZ = timezone(timedelta(hours=7))


def _nth_sunday_day(year, month, nth):
    first_day = datetime(year, month, 1, tzinfo=timezone.utc)
    days_until_sunday = (6 - first_day.weekday()) % 7
    return 1 + days_until_sunday + (nth - 1) * 7


def _is_us_dst_utc(dt_utc):
    dst_start = datetime(dt_utc.year, 3, _nth_sunday_day(dt_utc.year, 3, 2), 7, 0, tzinfo=timezone.utc)
    dst_end = datetime(dt_utc.year, 11, _nth_sunday_day(dt_utc.year, 11, 1), 6, 0, tzinfo=timezone.utc)
    return dst_start <= dt_utc < dst_end


def to_ny_time(dt_val):
    dt_utc = to_utc_datetime(dt_val, default_tz=VN_TZ)
    if ZoneInfo is not None:
        try:
            return dt_utc.astimezone(ZoneInfo("America/New_York"))
        except ZoneInfoNotFoundError:
            pass
    offset_hours = -4 if _is_us_dst_utc(dt_utc) else -5
    return dt_utc.astimezone(timezone(timedelta(hours=offset_hours)))


def to_utc_datetime(val, default_tz=timezone.utc):
    if val is None:
        return None
    if hasattr(val, "to_pydatetime"):
        val = val.to_pydatetime()
    if isinstance(val, datetime):
        aware = val if val.tzinfo is not None else val.replace(tzinfo=default_tz)
        return aware.astimezone(timezone.utc)
    text = str(val).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
        aware = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=default_tz)
        return aware.astimezone(timezone.utc)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(text[:len(fmt)], fmt).replace(tzinfo=default_tz).astimezone(timezone.utc)
            except ValueError:
                continue
    return None


def to_vn_datetime(val, default_tz=timezone.utc):
    dt_utc = to_utc_datetime(val, default_tz=default_tz)
    return dt_utc.astimezone(VN_TZ) if dt_utc is not None else None


def timeframe_to_timedelta(timeframe):
    tf = str(timeframe or "15m").strip().lower()
    if tf.endswith("m") and tf[:-1].isdigit():
        return timedelta(minutes=int(tf[:-1]))
    if tf.endswith("h") and tf[:-1].isdigit():
        return timedelta(hours=int(tf[:-1]))
    if tf.endswith("d") and tf[:-1].isdigit():
        return timedelta(days=int(tf[:-1]))
    return timedelta(minutes=15)


def format_timeframe_label(timeframe):
    tf = str(timeframe or "15m").strip().lower()
    if tf.endswith("m") and tf[:-1].isdigit():
        return f"M{tf[:-1]}"
    if tf.endswith("h") and tf[:-1].isdigit():
        return f"H{tf[:-1]}"
    if tf.endswith("d") and tf[:-1].isdigit():
        return f"D{tf[:-1]}"
    return tf.upper()


def is_market_open(symbol_name, now_vn):
    if symbol_name != "XAUUSD":
        return True, "Thị trường Crypto mở cửa 24/7"

    vn_dt = to_vn_datetime(now_vn, default_tz=VN_TZ)
    ny_dt = to_ny_time(vn_dt)
    close_vn_hour = 4 if ny_dt.utcoffset() == timedelta(hours=-4) else 5
    open_vn_hour = close_vn_hour + 1

    ny_wd, ny_hr = ny_dt.weekday(), ny_dt.hour
    if ny_wd == 4 and ny_hr >= 17:
        return False, f"Thị trường Vàng đã đóng cửa cuối tuần (từ {close_vn_hour:02d}:00 sáng Thứ Bảy VN). Mở lại {open_vn_hour:02d}:00 sáng Thứ Hai!"
    if ny_wd == 5:
        return False, f"Thị trường Vàng đang đóng cửa cuối tuần. Mở lại lúc {open_vn_hour:02d}:00 sáng Thứ Hai VN!"
    if ny_wd == 6 and ny_hr < 18:
        return False, f"Thị trường Vàng chưa mở cửa tuần mới (Mở lúc {open_vn_hour:02d}:00 sáng Thứ Hai VN)."
    if ny_wd in (0, 1, 2, 3) and ny_hr == 17:
        return False, f"Giờ nghỉ kỹ thuật hằng ngày của Vàng thế giới ({close_vn_hour:02d}:00 - {open_vn_hour:02d}:00 sáng VN)."
    return True, "Thị trường Vàng đang mở cửa giao dịch bình thường"


def build_future_timestamps(last_bar_time, symbol_name, timeframe, pred_len):
    step = timeframe_to_timedelta(timeframe)
    cursor = to_utc_datetime(last_bar_time, default_tz=timezone.utc) + step
    future_utc = []
    guard = 0
    while len(future_utc) < pred_len and guard < pred_len * 200:
        if is_market_open(symbol_name, cursor.astimezone(VN_TZ))[0]:
            future_utc.append(cursor)
        cursor = cursor + step
        guard += 1
    return future_utc


def format_forecast_window_vn(future_utc, timeframe):
    if not future_utc:
        return ""
    step = timeframe_to_timedelta(timeframe)
    start_vn = future_utc[0].astimezone(VN_TZ)
    end_vn = (future_utc[-1] + step).astimezone(VN_TZ)
    return f"{start_vn.strftime('%H:%M')} → {end_vn.strftime('%H:%M')} VN"
