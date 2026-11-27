import json
import os
import tempfile

STATE_FILE = os.path.join(os.path.dirname(__file__), "bot_state.json")


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as exc:
        print(f"[WARN] Could not read state file: {exc}")
        return {}


def save_state(state):
    state_dir = os.path.dirname(os.path.abspath(STATE_FILE))
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=state_dir,
            prefix=".bot_state_",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = tmp.name
            json.dump(state, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, STATE_FILE)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def has_sent_report(report_key, date_str):
    state = load_state()
    return state.get(report_key) == date_str


def mark_report_sent(report_key, date_str):
    state = load_state()
    state[report_key] = date_str
    save_state(state)
