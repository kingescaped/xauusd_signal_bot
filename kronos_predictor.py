from dataclasses import dataclass
import os
import sys
import numpy as np
import pandas as pd
import config
from kronos_loader import load_predictor_for_target, reset_loader_cache, resolve_hf_token
from market_time import build_future_timestamps, format_forecast_window_vn, to_utc_datetime
from meta_model import evaluate_meta_signal

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

_GLOBAL_PREDICTOR = None
_MODEL_LOADED = False


@dataclass(frozen=True)
class SymbolSpec:
    name: str
    timeframe: str
    digits: int
    threshold: float
    atr_fallback: float
    model_revision: str = "main"


@dataclass(frozen=True)
class MarketContext:
    ref_close: float
    entry_price: float
    atr: float
    sl_dist: float
    min_tp_dist: float
    adx: float
    ema_50: float
    ema_200: float


@dataclass(frozen=True)
class RawForecast:
    pred_end: float
    pred_high: float
    pred_low: float
    pct_change: float
    cand_direction: str
    confidence: float
    agreement_pct: float
    std_pct: float
    forecast_window_vn: str


def _resolve_hf_token():
    return resolve_hf_token()


def load_kronos_base(symbol_name=None, revision=None):
    global _GLOBAL_PREDICTOR, _MODEL_LOADED
    if _GLOBAL_PREDICTOR is None:
        reset_loader_cache()
    elif symbol_name is None and revision is None:
        return _GLOBAL_PREDICTOR

    predictor = load_predictor_for_target(symbol_name=symbol_name, revision=revision)
    if predictor is not None:
        _GLOBAL_PREDICTOR = predictor
        _MODEL_LOADED = True
    else:
        _MODEL_LOADED = False
    return predictor


def _get_symbol_spec(symbol_name, timeframe):
    is_gold = timeframe.lower() == "1h" or symbol_name == "XAUUSD"
    default_thresh = getattr(config, "GOLD_THRESHOLD", 0.30) if is_gold else getattr(config, "BTC_THRESHOLD", 0.55)
    default_atr = 5.0 if is_gold else 150.0
    default_rev = "xau-h1" if is_gold else "btc-m15"
    for item in getattr(config, "SYMBOLS", []):
        if item.get("name") == symbol_name:
            return SymbolSpec(
                name=symbol_name,
                timeframe=timeframe,
                digits=int(item.get("digits", 2)),
                threshold=float(item.get("threshold", default_thresh)),
                atr_fallback=float(item.get("atr_fallback", default_atr)),
                model_revision=str(item.get("model_revision") or default_rev),
            )
    return SymbolSpec(symbol_name, timeframe, 2, float(default_thresh), default_atr, default_rev)


def calculate_volatility(df, window=14, digits=2, fallback=5.0):
    if df is None or len(df) == 0:
        return fallback
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1.0 / window, adjust=False).mean().iloc[-1])
    if np.isnan(atr) or atr <= 0:
        return fallback
    rounded = round(atr, digits)
    return rounded if rounded > 0 else round(atr, max(digits, 6))


def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def calculate_adx(df, window=14):
    try:
        if df is None or len(df) < window * 2:
            return 0.0
        high, low, close = df["high"], df["low"], df["close"]
        up_move, down_move = high.diff(), -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        prev_c = close.shift(1)
        tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / window, adjust=False).mean().replace(0, np.nan)
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / window, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / window, adjust=False).mean() / atr)

        denom = (plus_di + minus_di).replace(0, np.nan)
        dx = (100 * ((plus_di - minus_di).abs() / denom)).fillna(0.0)
        adx = dx.ewm(alpha=1.0 / window, adjust=False).mean().iloc[-1]
        return round(float(adx), 2) if not np.isnan(adx) else 0.0
    except Exception:
        return 0.0


def _clean_candle_df(df):
    if df is None or getattr(df, "empty", True):
        return df
    cleaned = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if "volume" in cleaned.columns:
        cleaned["volume"] = cleaned["volume"].fillna(0.0)
    return cleaned


def _to_utc_series(time_col):
    if hasattr(time_col, "dt") and getattr(time_col.dt, "tz", None) is not None:
        return time_col.dt.tz_convert("UTC")
    raw_list = [time_col.iloc[i] for i in range(len(time_col))]
    utc_list = [to_utc_datetime(v) for v in raw_list if v is not None]
    parsed = pd.to_datetime([dt.isoformat() for dt in utc_list])
    if isinstance(pd.DatetimeIndex, type) and pd.DatetimeIndex is not tuple and isinstance(parsed, pd.DatetimeIndex):
        return pd.Series(parsed, index=getattr(time_col, "index", None))
    return parsed


def _prepare_kronos_inputs(clean_df, spec, pred_len):
    lookback = min(len(clean_df), getattr(config, "CANDLE_LIMIT", 200))
    context_df = clean_df.tail(lookback).copy()
    x_timestamp = _to_utc_series(context_df["time"])
    last_x_time = x_timestamp.iloc[-1] if hasattr(x_timestamp, "iloc") else x_timestamp[-1]
    future_utc = build_future_timestamps(last_x_time, spec.name, spec.timeframe, pred_len)
    y_timestamp = pd.DatetimeIndex(future_utc) if isinstance(pd.DatetimeIndex, type) and pd.DatetimeIndex is not tuple else future_utc
    window_vn = format_forecast_window_vn(future_utc, spec.timeframe)

    x_df = context_df[["open", "high", "low", "close", "volume"]].copy()
    if "amount" not in x_df.columns:
        x_df["amount"] = x_df["volume"] * x_df[["open", "high", "low", "close"]].mean(axis=1)
    return x_df, x_timestamp, y_timestamp, window_vn


def _run_kronos_predict(predictor, x_df, x_timestamp, y_timestamp, pred_len):
    kwargs = {
        "df": x_df, "x_timestamp": x_timestamp, "y_timestamp": y_timestamp, "pred_len": pred_len,
        "T": getattr(config, "KRONOS_TEMPERATURE", 0.6), "top_p": getattr(config, "KRONOS_TOP_P", 0.9),
        "sample_count": getattr(config, "KRONOS_SAMPLE_COUNT", 5), "verbose": False,
    }
    try:
        res = predictor.predict(**kwargs, return_samples=True)
    except TypeError:
        res = predictor.predict(**kwargs)
    return res if isinstance(res, tuple) and len(res) == 2 else (res, [res])


def _compute_mc_confidence(cand_direction, pct_change, ref_close, sample_dfs, threshold=0.30):
    if "SIDEWAY" in cand_direction or ref_close <= 0:
        return 50.0, 0.0, 0.0
    sample_pcts = [(float(s["close"].iloc[-1]) - ref_close) / ref_close * 100.0 for s in (sample_dfs or [])] or [pct_change]
    min_move = max(0.05, 0.5 * float(threshold))
    aligned = sum(1 for p in sample_pcts if (p >= min_move if "BUY" in cand_direction else p <= -min_move))
    agreement_ratio = aligned / len(sample_pcts)
    std_pct = float(np.std(sample_pcts))
    snr_score = min(1.0, (abs(pct_change) / (std_pct + 0.15)) / 2.5)
    dispersion_penalty = max(0.0, (std_pct / max(abs(pct_change), 0.1) - 0.8) * 12.0)
    raw_conf = 42.0 + 38.0 * agreement_ratio + 15.0 * snr_score - dispersion_penalty
    conf = round(float(np.clip(raw_conf, 35.0, 95.0)), 1)
    return conf, round(agreement_ratio * 100.0, 1), round(std_pct, 3)


def _summarize_samples(pred_df, sample_dfs, ref_close, spec, window_vn):
    if sample_dfs and len(sample_dfs) > 1:
        pred_end = float(np.median([float(s["close"].iloc[-1]) for s in sample_dfs]))
        pred_high = float(np.median([float(s["high"].max()) for s in sample_dfs]))
        pred_low = float(np.median([float(s["low"].min()) for s in sample_dfs]))
    else:
        pred_end = float(pred_df["close"].iloc[-1])
        pred_high, pred_low = float(pred_df["high"].max()), float(pred_df["low"].min())

    pct_change = (pred_end - ref_close) / ref_close * 100.0
    cand = "🟢 BUY (TĂNG)" if pct_change >= spec.threshold else ("🔴 SELL (GIẢM)" if pct_change <= -spec.threshold else "⚪ SIDEWAY (ĐI NGANG)")
    conf, agree_pct, std_pct = _compute_mc_confidence(cand, pct_change, ref_close, sample_dfs, spec.threshold)
    return RawForecast(pred_end, pred_high, pred_low, pct_change, cand, conf, agree_pct, std_pct, window_vn)


def _evaluate_entry_and_targets(spec, ctx, forecast):
    if "SIDEWAY" in forecast.cand_direction:
        return forecast.cand_direction, forecast.confidence, ctx.entry_price, ctx.entry_price, False

    is_buy = "BUY" in forecast.cand_direction
    signed_drift = (ctx.entry_price - ctx.ref_close) if is_buy else (ctx.ref_close - ctx.entry_price)
    max_adverse = getattr(config, "MAX_ENTRY_SLIPPAGE_ATR", 0.35) * ctx.atr
    max_pullback = getattr(config, "MAX_FAVORABLE_PULLBACK_ATR", 0.80) * ctx.atr
    d = spec.digits

    if signed_drift > max_adverse or signed_drift < -max_pullback:
        print(
            f"[BỘ LỌC TRƯỢT GIÁ] {spec.name}: Giá vào ({ctx.entry_price:.{d}f}) lệch {signed_drift:+.{d}f} "
            f"vượt ngưỡng cho phép so với nến đóng ({ctx.ref_close:.{d}f}, ATR={ctx.atr:.{d}f}) -> Bỏ kèo!"
        )
        return "⚪ SIDEWAY (ĐI NGANG)", 51.0, ctx.entry_price, ctx.entry_price, True

    tp = forecast.pred_high if is_buy else forecast.pred_low
    sl = (ctx.entry_price - ctx.sl_dist) if is_buy else (ctx.entry_price + ctx.sl_dist)
    reward = (tp - ctx.entry_price) if is_buy else (ctx.entry_price - tp)
    if reward < ctx.min_tp_dist - 1e-9:
        print(
            f"[BỘ LỌC R:R] {spec.name}: Biên độ TP dự phóng ({reward:.{d}f}) < chuẩn R:R tối thiểu "
            f"({ctx.min_tp_dist:.{d}f}) -> Bỏ kèo!"
        )
        return "⚪ SIDEWAY (ĐI NGANG)", 53.0, ctx.entry_price, ctx.entry_price, False
    return forecast.cand_direction, forecast.confidence, tp, sl, False


def _apply_regime_filters(spec, ctx, forecast):
    if getattr(config, "USE_SIDEWAY_FILTER", True) and ctx.adx < getattr(config, "ADX_THRESHOLD", 20.0):
        print(f"[BỘ LỌC SIDEWAY] {spec.name}: ADX = {ctx.adx:.1f} -> Thị trường đi ngang tích lũy, giữ im lặng.")
        return "⚪ SIDEWAY (ĐI NGANG)", 50.0, ctx.entry_price, ctx.entry_price, False

    if getattr(config, "USE_TREND_FILTER", True):
        if forecast.cand_direction == "🟢 BUY (TĂNG)" and (ctx.ref_close < ctx.ema_50 or ctx.ref_close < ctx.ema_200):
            print(f"[BỘ LỌC XU HƯỚNG EMA] {spec.name}: Giá ({ctx.ref_close:.{spec.digits}f}) < EMA50/200 ({ctx.ema_50:.1f}/{ctx.ema_200:.1f}) -> Bỏ qua BUY")
            return "⚪ SIDEWAY (ĐI NGANG)", 52.0, ctx.entry_price, ctx.entry_price, False
        if forecast.cand_direction == "🔴 SELL (GIẢM)" and (ctx.ref_close > ctx.ema_50 or ctx.ref_close > ctx.ema_200):
            print(f"[BỘ LỌC XU HƯỚNG EMA] {spec.name}: Giá ({ctx.ref_close:.{spec.digits}f}) > EMA50/200 ({ctx.ema_50:.1f}/{ctx.ema_200:.1f}) -> Bỏ qua SELL")
            return "⚪ SIDEWAY (ĐI NGANG)", 52.0, ctx.entry_price, ctx.entry_price, False

    return _evaluate_entry_and_targets(spec, ctx, forecast)


def _build_no_ai_response(spec, ctx):
    d = spec.digits
    return {
        "current_price": round(ctx.entry_price, d), "ref_close": round(ctx.ref_close, d),
        "direction": "⚪ SIDEWAY (ĐI NGANG)", "confidence": 0.0,
        "tp": round(ctx.entry_price, d), "sl": round(ctx.entry_price, d),
        "atr": ctx.atr, "ai_predicted": False, "slippage_blocked": False, "pct_change": 0.0,
        "pred_end_price": round(ctx.ref_close, d), "pred_high": round(ctx.ref_close, d), "pred_low": round(ctx.ref_close, d),
        "adx": ctx.adx, "ema_50": round(ctx.ema_50, d), "ema_200": round(ctx.ema_200, d),
        "sample_agreement": 0.0, "sample_std_pct": 0.0, "forecast_window_vn": "",
        "strategy_version": getattr(config, "STRATEGY_VERSION", "v4.0-dual-kronos-meta"),
    }


def _build_market_context(clean_df, spec, live_price):
    ref_close = float(clean_df["close"].iloc[-1])
    entry_price = float(live_price) if (live_price is not None and not np.isnan(live_price)) else ref_close
    atr = calculate_volatility(clean_df, digits=spec.digits, fallback=spec.atr_fallback)
    sl_dist = atr * getattr(config, "ATR_SL_MULTIPLIER", 1.5)
    min_tp_dist = sl_dist * getattr(config, "MIN_RR_RATIO", 1.2)
    adx_val = calculate_adx(clean_df)
    ema_50 = float(calculate_ema(clean_df["close"], 50).iloc[-1])
    ema_200 = float(calculate_ema(clean_df["close"], 200).iloc[-1])
    return MarketContext(ref_close, entry_price, atr, sl_dist, min_tp_dist, adx_val, ema_50, ema_200)


def _attach_meta_score(symbol_name, payload):
    if "SIDEWAY" in payload["direction"]:
        return payload
    try:
        from history_tracker import load_history
        hist = load_history()
        open_cnt = sum(1 for s in hist if s.get("symbol") == symbol_name and s.get("status") == "OPEN")
    except Exception:
        hist, open_cnt = None, 0
    decision = evaluate_meta_signal(symbol_name, payload, history=hist, order_num=open_cnt + 1)
    payload["meta_win_prob"] = round(decision.win_prob * 100.0, 1)
    payload["meta_filter_active"] = decision.filter_active
    return payload


def predict_future(df, symbol_name, timeframe="15m", pred_len=12, live_price=None):
    spec = _get_symbol_spec(symbol_name, timeframe)
    clean_df = _clean_candle_df(df)
    if clean_df is None or clean_df.empty:
        empty_ctx = MarketContext(0.0, 0.0, spec.atr_fallback, 0.0, 0.0, 0.0, 0.0, 0.0)
        return _build_no_ai_response(spec, empty_ctx)

    ctx = _build_market_context(clean_df, spec, live_price)
    predictor = load_kronos_base(spec.name, spec.model_revision)
    if predictor is not None and len(clean_df) >= 25:
        try:
            print(f"[AI KRONOS] Đang chạy suy luận K-line cho {symbol_name} ({timeframe})...")
            x_df, x_timestamp, y_timestamp, window_vn = _prepare_kronos_inputs(clean_df, spec, pred_len)
            pred_df, sample_dfs = _run_kronos_predict(predictor, x_df, x_timestamp, y_timestamp, pred_len)
            forecast = _summarize_samples(pred_df, sample_dfs, ctx.ref_close, spec, window_vn)
            direction, conf, tp, sl, slip_blocked = _apply_regime_filters(spec, ctx, forecast)
            d = spec.digits
            print(
                f"[AI KRONOS PHÂN TÍCH] {symbol_name} ({timeframe}): {direction} | Dự phóng: {forecast.pct_change:+.2f}% "
                f"| Đồng thuận: {forecast.agreement_pct}% | ADX: {ctx.adx:.1f} | Tin cậy: {conf}%"
            )
            model_src = getattr(predictor, "model_source", f"{config.KRONOS_MODEL_NAME}@{spec.model_revision}")
            payload = {
                "current_price": round(ctx.entry_price, d), "ref_close": round(ctx.ref_close, d),
                "direction": direction, "confidence": conf, "tp": round(tp, d), "sl": round(sl, d),
                "atr": ctx.atr, "ai_predicted": True, "slippage_blocked": slip_blocked,
                "pct_change": round(forecast.pct_change, 2), "pred_end_price": round(forecast.pred_end, d),
                "pred_high": round(forecast.pred_high, d), "pred_low": round(forecast.pred_low, d),
                "adx": ctx.adx, "ema_50": round(ctx.ema_50, d), "ema_200": round(ctx.ema_200, d),
                "sample_agreement": forecast.agreement_pct, "sample_std_pct": forecast.std_pct,
                "forecast_window_vn": forecast.forecast_window_vn, "model_source": model_src,
                "strategy_version": getattr(config, "STRATEGY_VERSION", "v4.0-dual-kronos-meta"),
            }
            return _attach_meta_score(symbol_name, payload)
        except Exception as e:
            print(f"[CẢNH BÁO AI] Lỗi trong quá trình suy luận Kronos: {e}")

    return _build_no_ai_response(spec, ctx)
