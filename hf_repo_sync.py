import os
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
    main_uploaded = _copy_missing_files_to_branch(api, resolver, target_repo, "main")

    branch_summary = {"main": main_uploaded}
    for branch in branches:
        api.create_branch(repo_id=target_repo, repo_type="model", branch=branch, revision="main", exist_ok=True)
        copied = _copy_missing_files_to_branch(api, resolver, target_repo, branch)
        branch_summary[branch] = copied

    print(f"[HF SYNC] Đã đồng bộ Private repo {target_repo} thành công: {branch_summary}")
    return {"repo_id": target_repo, "private": True, "branches": branch_summary}
