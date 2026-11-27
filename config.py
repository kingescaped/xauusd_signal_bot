import os


def _load_local_dotenv(env_path=None):
    env_path = env_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8-sig") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                    val = val[1:-1]
                if key:
                    os.environ.setdefault(key, val)
    except OSError:
        pass


_load_local_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_GROUP_ID = os.getenv("TELEGRAM_GROUP_ID", "")

KRONOS_MODEL_NAME = os.getenv("KRONOS_MODEL_NAME", "kingescaped/lvq-std-BOT")
KRONOS_FALLBACK_MODEL_NAME = os.getenv("KRONOS_FALLBACK_MODEL_NAME", "NeoQuasar/Kronos-base")
KRONOS_TOKENIZER_NAME = os.getenv("KRONOS_TOKENIZER_NAME", "NeoQuasar/Kronos-Tokenizer-base")
hf_token_val = os.getenv("HF_TOKEN", "").strip()
HF_TOKEN = hf_token_val if hf_token_val else None
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
else:
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

SYMBOLS = [
    {
        "name": "BTCUSD",
        "ticker": "BTC-USD",
        "label": "BITCOIN (BTCUSD)",
        "timeframe": "15m",
        "model_revision": "btc-m15",
        "unit": "$",
        "digits": 2,
        "threshold": 0.55,
        "atr_fallback": 150.0,
        "predict_len": 12,
        "period": "5d",
    },
    {
        "name": "XAUUSD",
        "ticker": "GC=F",
        "label": "VÀNG (XAUUSD)",
        "timeframe": "1h",
        "model_revision": "xau-h1",
        "unit": "$",
        "digits": 2,
        "threshold": 0.30,
        "atr_fallback": 5.0,
        "predict_len": 6,
        "period": "30d",
    },
]

CANDLE_LIMIT = 200
INDICATOR_CANDLE_LIMIT = 450

CONFIDENCE_THRESHOLD = 75.0
BTC_THRESHOLD = 0.55
GOLD_THRESHOLD = 0.30

KRONOS_SAMPLE_COUNT = 5
KRONOS_TEMPERATURE = 0.6
KRONOS_TOP_P = 0.9
STRATEGY_VERSION = "v4.0-dual-kronos-meta"

META_MIN_CLEAN_TRADES = 100
META_WIN_PROB_THRESHOLD = 0.52
META_CLEAN_VERSION_PREFIXES = ("v3.3", "v4.")

ATR_SL_MULTIPLIER = 1.5
MIN_RR_RATIO = 1.2

ALLOW_OPPOSITE_HEDGING = False
MAX_OPEN_SIGNALS_PER_SYMBOL = 2
MIN_ENTRY_DISTANCE_ATR = 0.8
MAX_ENTRY_SLIPPAGE_ATR = 0.35
MAX_FAVORABLE_PULLBACK_ATR = 0.80

USE_TREND_FILTER = True
USE_SIDEWAY_FILTER = True
ADX_THRESHOLD = 20.0

DAILY_REPORT_HOUR_VN = 6
