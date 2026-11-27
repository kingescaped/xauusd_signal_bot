import json
import os
import shutil
import tempfile
import config
from kronos_loader import resolve_hf_token

REQUIRED_FILES = ("config.json", "model.safetensors", "README.md")
DEFAULT_BRANCHES = ("btc-m15", "xau-h1")


def _list_branch_files(api, repo_id, revision):
    try:
        return set(api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision))
    except Exception:
        return set()


def _resolve_source_file(download_fn, source_repo, filename, local_dir=None):
    if local_dir:
        candidate = os.path.join(local_dir, filename)
        if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            return candidate
    return download_fn(repo_id=source_repo, filename=filename, repo_type="model", token=False)


def _copy_missing_files_to_branch(api, resolver_fn, target_repo, revision):
    existing = _list_branch_files(api, target_repo, revision)
    uploaded = []
    for filename in REQUIRED_FILES:
        if filename in existing:
            continue
        local_path = resolver_fn(filename)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=target_repo,
            repo_type="model",
            revision=revision,
            commit_message=f"Sync {filename} to {revision}",
        )
        uploaded.append(filename)
    return uploaded


def _is_tokenizer_branch_ready(api, download_fn, target_repo, tok_branch, token):
    if not set(REQUIRED_FILES).issubset(_list_branch_files(api, target_repo, tok_branch)):
        return False
    try:
        cfg_path = download_fn(repo_id=target_repo, filename="config.json", repo_type="model", revision=tok_branch, token=token)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        return "d_in" in cfg_data and "n_enc_layers" in cfg_data
    except Exception:
        return False


def _sync_tokenizer_branch(api, download_fn, target_repo, token):
    tok_branch = getattr(config, "KRONOS_TOKENIZER_REVISION", "tokenizer") or "tokenizer"
    source_tok = getattr(config, "KRONOS_FALLBACK_TOKENIZER_NAME", "NeoQuasar/Kronos-Tokenizer-base")
    api.create_branch(repo_id=target_repo, repo_type="model", branch=tok_branch, revision="main", exist_ok=True)
    if _is_tokenizer_branch_ready(api, download_fn, target_repo, tok_branch, token):
        return []
    with tempfile.TemporaryDirectory() as stage_dir:
        for fname in REQUIRED_FILES:
            src_file = download_fn(repo_id=source_tok, filename=fname, repo_type="model", token=False)
            shutil.copyfile(src_file, os.path.join(stage_dir, fname))
        api.upload_folder(
            folder_path=stage_dir,
            repo_id=target_repo,
            repo_type="model",
            revision=tok_branch,
            commit_message=f"Sync KronosTokenizer from {source_tok} to {tok_branch}",
        )
    return list(REQUIRED_FILES)


def sync_base_to_private_repo(target_repo=None, source_repo=None, branches=DEFAULT_BRANCHES, token=None, local_dir=None):
    hf_token = (token or resolve_hf_token() or os.getenv("HF_TOKEN") or "").strip()
    if not hf_token:
        raise ValueError(
            "Missing HF_TOKEN with Write permission. Create one at https://huggingface.co/settings/tokens "
            "and set HF_TOKEN in GitHub Secrets / Colab Secrets / .env."
        )

    from huggingface_hub import HfApi, hf_hub_download

    target_repo = target_repo or getattr(config, "KRONOS_MODEL_NAME", "kingescaped/lvq-std-BOT")
    source_repo = source_repo or getattr(config, "KRONOS_FALLBACK_MODEL_NAME", "NeoQuasar/Kronos-base")
    api = HfApi(token=hf_token)
    resolver = lambda fname: _resolve_source_file(hf_hub_download, source_repo, fname, local_dir=local_dir)

    api.create_repo(repo_id=target_repo, repo_type="model", private=True, exist_ok=True)
    branch_summary = {"main": _copy_missing_files_to_branch(api, resolver, target_repo, "main")}
    for branch in branches:
        api.create_branch(repo_id=target_repo, repo_type="model", branch=branch, revision="main", exist_ok=True)
        branch_summary[branch] = _copy_missing_files_to_branch(api, resolver, target_repo, branch)
    tok_branch = getattr(config, "KRONOS_TOKENIZER_REVISION", "tokenizer")
    branch_summary[tok_branch] = _sync_tokenizer_branch(api, hf_hub_download, target_repo, hf_token)

    print(f"[HF SYNC] Đã đồng bộ Private repo {target_repo} thành công: {branch_summary}")
    return {"repo_id": target_repo, "private": True, "branches": branch_summary}
