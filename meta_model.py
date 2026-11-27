from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
import tempfile
import numpy as np
import config

_DEFAULT_META_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta_model_state.json")
META_STATE_FILE = _DEFAULT_META_STATE_FILE


def _resolve_state_path():
    if META_STATE_FILE != _DEFAULT_META_STATE_FILE:
        return META_STATE_FILE
    try:
        import history_tracker
        hist_dir = os.path.dirname(os.path.abspath(history_tracker.HISTORY_FILE))
        return os.path.join(hist_dir, "meta_model_state.json")
    except Exception:
        return META_STATE_FILE
FEATURE_NAMES = (
    "is_gold",
    "is_buy",
    "confidence_norm",
    "abs_pct_change",
    "agreement_norm",
    "sample_std_pct",
    "adx_norm",
    "rr_ratio",
    "ema50_dist_atr",
    "is_dca",
)


@dataclass(frozen=True)
class MetaDecision:
    win_prob: float
    clean_trades: int
    filter_active: bool
    blocked: bool
    reason: str = ""


def _is_valid_num(val):
    return isinstance(val, (int, float)) and not isinstance(val, bool) and not math.isnan(float(val))


def is_clean_closed_signal(sig):
    if not isinstance(sig, dict) or sig.get("status") not in ("HIT_TP", "HIT_SL"):
        return False
    version = str(sig.get("strategy_version") or "")
    prefixes = getattr(config, "META_CLEAN_VERSION_PREFIXES", ("v3.3", "v4."))
    if not version or not any(version.startswith(p) for p in prefixes):
        return False
    required = ("entry_price", "tp", "sl", "confidence", "pct_change", "atr", "adx", "sample_agreement", "sample_std_pct")
    if not all(_is_valid_num(sig.get(k)) for k in required):
        return False
    return float(sig["atr"]) > 0 and abs(float(sig["entry_price"]) - float(sig["sl"])) > 0


def _safe_num(record, key, default):
    val = record.get(key) if isinstance(record, dict) else None
    return float(val) if _is_valid_num(val) else float(default)


def extract_meta_features(symbol, record, order_num=1):
    fallback_entry = _safe_num(record, "current_price", 0.0)
    entry = _safe_num(record, "entry_price", fallback_entry)
    tp = _safe_num(record, "tp", entry)
    sl = _safe_num(record, "sl", entry)
    raw_atr = _safe_num(record, "atr", 1.0)
    atr = raw_atr if raw_atr > 1e-6 else 1.0
    ref_close = _safe_num(record, "ref_close", entry)
    ema_50 = _safe_num(record, "ema_50", ref_close)
    is_buy = 1.0 if "BUY" in str(record.get("direction", "")) else 0.0
    sign = 1.0 if is_buy > 0.5 else -1.0

    risk = max(abs(entry - sl), 1e-6)
    reward = max(sign * (tp - entry), 0.0)
    rr_ratio = float(np.clip(reward / risk, 0.0, 5.0))
    ema_dist = float(np.clip(sign * (ref_close - ema_50) / atr, -5.0, 5.0))
    ord_val = int(_safe_num(record, "order_num", order_num))
    dca_flag = 1.0 if (bool(record.get("is_dca")) or ord_val > 1) else 0.0

    return np.array([
        1.0 if symbol == "XAUUSD" else 0.0,
        is_buy,
        (_safe_num(record, "confidence", 75.0) - 75.0) / 20.0,
        abs(_safe_num(record, "pct_change", 0.0)),
        _safe_num(record, "sample_agreement", 80.0) / 100.0,
        _safe_num(record, "sample_std_pct", 0.2),
        _safe_num(record, "adx", 25.0) / 50.0,
        rr_ratio,
        ema_dist,
        dca_flag,
    ], dtype=float)


def _sigmoid(z):
    clipped = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logistic_l2(x_mat, y_vec, l2_penalty=1.0, max_iter=60):
    means = np.mean(x_mat, axis=0)
    stds = np.where(np.std(x_mat, axis=0) < 1e-6, 1.0, np.std(x_mat, axis=0))
    x_norm = (x_mat - means) / stds
    n_samples, n_feat = x_norm.shape
    design = np.column_stack([np.ones(n_samples), x_norm])
    weights = np.zeros(n_feat + 1, dtype=float)
    reg = np.diag([0.0] + [float(l2_penalty)] * n_feat)

    for _ in range(max_iter):
        probs = _sigmoid(design @ weights)
        grad = design.T @ (probs - y_vec) + reg @ weights
        w_diag = np.clip(probs * (1.0 - probs), 1e-5, None)
        hess = (design.T * w_diag) @ design + reg
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        weights = weights - step
        if float(np.max(np.abs(step))) < 1e-6:
            break
    return float(weights[0]), weights[1:].tolist(), means.tolist(), stds.tolist()


def load_meta_state():
    state_path = _resolve_state_path()
    if not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_meta_state(state):
    state_path = _resolve_state_path()
    target_dir = os.path.dirname(os.path.abspath(state_path))
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target_dir, prefix=".meta_state_", suffix=".tmp", delete=False
        ) as tmp:
            tmp_path = tmp.name
            json.dump(state, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, state_path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def train_weekly_meta_model(history):
    min_trades = int(getattr(config, "META_MIN_CLEAN_TRADES", 100))
    threshold = float(getattr(config, "META_WIN_PROB_THRESHOLD", 0.52))
    clean_items = [s for s in (history or []) if is_clean_closed_signal(s)]
    clean_count = len(clean_items)
    y_list = [1.0 if s["status"] == "HIT_TP" else 0.0 for s in clean_items]
    has_both_classes = len(set(y_list)) >= 2

    state = {
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clean_trade_count": clean_count,
        "min_clean_trades": min_trades,
        "threshold": threshold,
        "feature_names": list(FEATURE_NAMES),
        "filter_active": bool(clean_count >= min_trades and has_both_classes),
        "trained": False,
    }
    if clean_count >= 20 and has_both_classes:
        x_mat = np.vstack([extract_meta_features(s.get("symbol", ""), s, s.get("order_num", 1)) for s in clean_items])
        y_vec = np.asarray(y_list, dtype=float)
        bias, coefs, means, stds = _fit_logistic_l2(x_mat, y_vec, l2_penalty=1.0)
        preds = _sigmoid(bias + ((x_mat - np.asarray(means)) / np.asarray(stds)) @ np.asarray(coefs))
        state.update({
            "trained": True,
            "bias": round(bias, 6),
            "weights": [round(w, 6) for w in coefs],
            "means": [round(m, 6) for m in means],
            "stds": [round(s, 6) for s in stds],
            "empirical_win_rate": round(float(np.mean(y_vec)), 4),
            "train_accuracy": round(float(np.mean((preds >= 0.5) == (y_vec >= 0.5))), 4),
        })
    save_meta_state(state)
    return state


def _heuristic_win_prob(feat):
    conf_norm, agree_norm, std_pct, adx_norm, rr_ratio = feat[2], feat[4], feat[5], feat[6], feat[7]
    logit = -0.15 + 0.45 * conf_norm + 0.35 * (agree_norm - 0.75) + 0.15 * (adx_norm - 0.5) + 0.08 * (rr_ratio - 1.2) - 0.10 * std_pct
    return float(np.clip(_sigmoid(logit), 0.05, 0.95))


def _score_with_state(feat, state):
    if state.get("trained") and len(state.get("weights", [])) == len(FEATURE_NAMES):
        means = np.asarray(state["means"], dtype=float)
        stds = np.where(np.asarray(state["stds"], dtype=float) < 1e-6, 1.0, np.asarray(state["stds"], dtype=float))
        weights = np.asarray(state["weights"], dtype=float)
        z = float(state.get("bias", 0.0)) + float(((feat - means) / stds) @ weights)
        return float(np.clip(_sigmoid(z), 0.01, 0.99))
    return _heuristic_win_prob(feat)


def evaluate_meta_signal(symbol, record, history=None, order_num=1):
    min_trades = int(getattr(config, "META_MIN_CLEAN_TRADES", 100))
    threshold = float(getattr(config, "META_WIN_PROB_THRESHOLD", 0.52))
    state = load_meta_state()
    if history is not None:
        clean_items = [s for s in history if is_clean_closed_signal(s)]
        clean_count = len(clean_items)
        has_both = len({s["status"] for s in clean_items}) >= 2
        if clean_count >= min_trades and has_both and not state.get("filter_active"):
            state = train_weekly_meta_model(history)
    else:
        clean_count = int(state.get("clean_trade_count", 0))

    feat = extract_meta_features(symbol, record, order_num=order_num)
    win_prob = _score_with_state(feat, state)
    filter_active = bool(clean_count >= min_trades and state.get("trained") and state.get("filter_active", False))
    blocked = bool(filter_active and win_prob < threshold)
    reason = ""
    if blocked:
        reason = f"Meta-Model chặn kèo: xác suất thắng {win_prob * 100:.1f}% < {threshold * 100:.0f}% ({clean_count} lệnh sạch)"
    return MetaDecision(round(win_prob, 4), clean_count, filter_active, blocked, reason)
