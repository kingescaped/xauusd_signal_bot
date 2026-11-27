import json
import os
import urllib.error
import urllib.parse
import urllib.request

KAGGLE_API_BASE = "https://www.kaggle.com/api/v1"
DEFAULT_KERNEL_TITLE = "AI LVQ Std BOT Monthly Fine Tune"
DEFAULT_KERNEL_SLUG = "ai-lvq-std-bot-monthly-fine-tune"
DEFAULT_NOTEBOOK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kaggle", "finetune_kaggle_monthly.ipynb"
)


def normalize_kaggle_token(raw_token):
    cleaned = str(raw_token or "").strip()
    if cleaned.startswith("KAGGLE_API_TOKEN="):
        cleaned = cleaned[len("KAGGLE_API_TOKEN="):].strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
        cleaned = cleaned[1:-1].strip()
    if not cleaned:
        raise ValueError(
            "Missing KAGGLE_API_TOKEN. Add KAGGLE_API_TOKEN (KGAT_...) to GitHub Actions Secrets."
        )
    return cleaned


def _kaggle_json_request(path, token, payload=None):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{KAGGLE_API_BASE}{path}", data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="ignore")[:300]
        finally:
            exc.close()
        raise RuntimeError(f"Kaggle API HTTP {exc.code} on {path}: {detail}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Kaggle API network error on {path}: {exc.reason}") from None


def fetch_kaggle_username(token):
    data = _kaggle_json_request("/hello", token)
    username = str((data or {}).get("userName") or "").strip()
    if not username:
        raise RuntimeError("Could not resolve Kaggle username from KAGGLE_API_TOKEN.")
    return username


def kernel_exists_for_user(username, kernel_slug, token):
    encoded_slug = urllib.parse.quote(kernel_slug)
    items = _kaggle_json_request(f"/kernels/list?group=profile&search={encoded_slug}&pageSize=50", token)
    target_ref = f"{username}/{kernel_slug}".lower()
    if not isinstance(items, list):
        return False
    return any(str(item.get("ref") or "").lstrip("/").removeprefix("code/").lower() == target_ref for item in items if isinstance(item, dict))


def build_push_payload(username, notebook_text, is_existing):
    payload = {
        "newTitle": DEFAULT_KERNEL_TITLE,
        "text": notebook_text,
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": True,
        "enableGpu": True,
        "enableTpu": False,
        "enableInternet": True,
        "machineShape": "NvidiaTeslaT4",
        "datasetDataSources": [],
        "competitionDataSources": [],
        "kernelDataSources": [],
    }
    if is_existing:
        payload["slug"] = f"{username}/{DEFAULT_KERNEL_SLUG}"
    return payload


def _read_validated_notebook(notebook_path):
    target_path = notebook_path or DEFAULT_NOTEBOOK_PATH
    with open(target_path, "r", encoding="utf-8") as nb_file:
        text = nb_file.read()
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or "cells" not in parsed:
        raise ValueError(f"Invalid Jupyter notebook structure in {target_path}")
    return text


def _submit_with_conflict_retry(username, notebook_text, exists, token):
    try:
        result = _kaggle_json_request("/kernels/push", token, payload=build_push_payload(username, notebook_text, exists))
    except RuntimeError as exc:
        if not exists and ("HTTP 409" in str(exc) or "already" in str(exc).lower()):
            return _kaggle_json_request("/kernels/push", token, payload=build_push_payload(username, notebook_text, True))
        raise
    err_msg = str(result.get("error") or "")
    if not exists and "already" in err_msg.lower():
        return _kaggle_json_request("/kernels/push", token, payload=build_push_payload(username, notebook_text, True))
    return result


def push_monthly_finetune_kernel(notebook_path=None, token=None):
    clean_token = normalize_kaggle_token(token or os.getenv("KAGGLE_API_TOKEN"))
    notebook_text = _read_validated_notebook(notebook_path)
    username = (os.getenv("KAGGLE_USERNAME") or "").strip() or fetch_kaggle_username(clean_token)
    exists = kernel_exists_for_user(username, DEFAULT_KERNEL_SLUG, clean_token)
    result = _submit_with_conflict_retry(username, notebook_text, exists, clean_token)
    if result.get("error"):
        raise RuntimeError(f"Kaggle kernel push rejected: {result['error']}")
    raw_ref = str(result.get("ref") or f"{username}/{DEFAULT_KERNEL_SLUG}").lstrip("/")
    ref = raw_ref[len("code/"):] if raw_ref.startswith("code/") else raw_ref
    ver = result.get("versionNumber", 1)
    url = result.get("url") or f"https://www.kaggle.com/code/{ref}"
    print(f"[AI LVQ Std BOT] Đã kích hoạt Kaggle GPU thành công: {ref} (v{ver}) -> {url}")
    return result


if __name__ == "__main__":
    push_monthly_finetune_kernel()
