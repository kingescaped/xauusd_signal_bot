import os
import time
import requests
import config

_PLACEHOLDERS = {"DIEN_BOT_TOKEN_VAO_DAY", "DIEN_GROUP_ID_VAO_DAY"}


def _resolve_bot_token(bot_token=None):
    raw = getattr(config, "TELEGRAM_BOT_TOKEN", "") if bot_token is None else bot_token
    token = str(raw or "").strip()
    return "" if token in _PLACEHOLDERS else token


def _resolve_target_chat_id(chat_id=None):
    raw = getattr(config, "TELEGRAM_GROUP_ID", "") if chat_id is None else chat_id
    target = str(raw or "").strip()
    return "" if target in _PLACEHOLDERS else target


def _has_valid_credentials(chat_id=None, bot_token=None):
    if not _resolve_bot_token(bot_token) or not _resolve_target_chat_id(chat_id):
        print("[LỖI] Thiếu TELEGRAM_BOT_TOKEN hoặc TELEGRAM_GROUP_ID!")
        return False
    return True


def _post_with_retry(request_fn, label, retries=3, base_delay=1.5):
    for attempt in range(1, retries + 1):
        try:
            response = request_fn()
            res_data = response.json()
            if not isinstance(res_data, dict):
                raise ValueError("Invalid Telegram JSON payload")
            if res_data.get("ok"):
                print(f"[THÀNH CÔNG] Đã {label} Telegram.")
                return True
            status = getattr(response, "status_code", 200)
            retryable = status == 429 or status >= 500
            if not retryable or attempt == retries:
                print(f"[THẤT BẠI] Lỗi Telegram API ({status}): {res_data.get('description')}")
                return False
            wait_sec = min(10.0, float((res_data.get("parameters") or {}).get("retry_after", base_delay * attempt)))
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                print(f"[LỖI] Kết nối Telegram thất bại sau {retries} lần: {exc}")
                return False
            wait_sec = base_delay * attempt
        time.sleep(wait_sec)
    return False


def send_telegram_message(text, chat_id=None, bot_token=None):
    if not _has_valid_credentials(chat_id=chat_id, bot_token=bot_token):
        return False
    token = _resolve_bot_token(bot_token)
    target_id = _resolve_target_chat_id(chat_id)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": target_id, "text": text, "parse_mode": "HTML"}
    label = "gửi tin nhắn riêng qua" if chat_id is not None else "gửi tin nhắn vào nhóm"
    return _post_with_retry(lambda: requests.post(url, json=payload, timeout=10), label)


def _match_actor_id(actor, clean_username):
    if not isinstance(actor, dict):
        return None
    uname = str(actor.get("username") or "").strip().lstrip("@").lower()
    if uname == clean_username and actor.get("id") is not None:
        return str(actor["id"])
    return None


def _match_block_user_id(item, clean_username):
    if not isinstance(item, dict):
        return None
    for key in ("message", "edited_message", "my_chat_member"):
        block = item.get(key) or {}
        matched = _match_actor_id(block.get("from"), clean_username) or _match_actor_id(block.get("chat"), clean_username)
        if matched:
            return matched
    return None


def _lookup_id_from_group_admins(token, group_id, clean_username):
    if not group_id:
        return None
    try:
        resp = requests.get(f"https://api.telegram.org/bot{token}/getChatAdministrators", params={"chat_id": group_id}, timeout=10)
        data = resp.json()
        for member in (data.get("result") or []) if isinstance(data, dict) and data.get("ok") else []:
            matched = _match_actor_id((member or {}).get("user"), clean_username)
            if matched:
                return matched
    except (requests.RequestException, ValueError):
        return None
    return None


def _lookup_id_from_updates(token, clean_username):
    try:
        resp = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", params={"offset": -100, "limit": 100}, timeout=10)
        data = resp.json()
        updates = (data.get("result") or []) if isinstance(data, dict) and data.get("ok") else []
        for item in reversed(updates):
            matched = _match_block_user_id(item, clean_username)
            if matched:
                return matched
    except (requests.RequestException, ValueError):
        return None
    return None


def resolve_personal_chat_id(target_username="kingescaped", explicit_chat_id=None, bot_token=None, group_id=None):
    explicit = str(explicit_chat_id or os.getenv("TELEGRAM_ADMIN_ID") or "").strip()
    if explicit and not explicit.startswith("@") and explicit not in _PLACEHOLDERS:
        return explicit
    token = _resolve_bot_token(bot_token)
    clean_user = str(target_username or "").strip().lstrip("@").lower()
    if not token or not clean_user:
        return None
    eff_group = _resolve_target_chat_id(group_id)
    return _lookup_id_from_group_admins(token, eff_group, clean_user) or _lookup_id_from_updates(token, clean_user)


def send_telegram_direct_message(text, target_username="kingescaped", chat_id=None, bot_token=None, group_id=None):
    personal_id = resolve_personal_chat_id(target_username, explicit_chat_id=chat_id, bot_token=bot_token, group_id=group_id)
    if not personal_id:
        print(
            f"[CẢNH BÁO TELEGRAM] Chưa tìm thấy User ID số của @{str(target_username).lstrip('@')}. "
            "Hãy bấm /start vào Bot trên Telegram và khai báo TELEGRAM_ADMIN_ID!"
        )
        return False
    return send_telegram_message(text, chat_id=personal_id, bot_token=bot_token)


def send_telegram_photo(photo_path, caption):
    if not _has_valid_credentials():
        return False
    if not photo_path or not os.path.exists(photo_path):
        return send_telegram_message(caption)
    token = _resolve_bot_token()
    target_id = _resolve_target_chat_id()
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    data = {"chat_id": target_id, "caption": caption, "parse_mode": "HTML"}

    def _send_file():
        with open(photo_path, "rb") as photo_file:
            return requests.post(url, data=data, files={"photo": photo_file}, timeout=20)

    if _post_with_retry(_send_file, "gửi ảnh biểu đồ vào nhóm"):
        return True
    print("[CẢNH BÁO] Gửi ảnh Telegram thất bại -> chuyển sang gửi tin nhắn văn bản dự phòng.")
    return send_telegram_message(caption)


def test_telegram():
    test_msg = (
        "🤖 <b>[VÀNG Đánh BTC] - KẾT NỐI THÀNH CÔNG!</b>\n\n"
        "Chào mừng bạn! Hệ thống tín hiệu AI LVQ Std BOT đã kết nối thành công với nhóm Telegram này.\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "📊 <b>Sản phẩm theo dõi:</b>\n"
        "🟡 <b>Vàng (XAUUSD):</b> Khung H1 (1 Giờ)\n"
        "🟠 <b>Bitcoin (BTCUSD):</b> Khung M15 (15 Phút)\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "⏰ <b>Lịch báo cáo:</b> 06:00 Sáng hàng ngày &amp; Thứ Bảy hàng tuần."
    )
    return send_telegram_message(test_msg)


if __name__ == "__main__":
    test_telegram()
