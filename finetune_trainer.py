import copy
from dataclasses import dataclass
import html
import math
import os
import tempfile
import torch
import config
from finetune_data import compute_recency_weights
from kronos_loader import load_tokenizer_instance, resolve_hf_token


@dataclass(frozen=True)
class FineTuneConfig:
    symbol: str
    branch: str
    context_len: int = 200
    pred_len: int = 12
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-6
    weight_decay: float = 0.1
    adam_betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 3.0
    stride: int = 4
    val_tolerance: float = 0.015
    anomaly_val_loss: float = 3.0
    anomaly_spike_delta: float = 0.35
    recency_min_weight: float = 0.5
    recency_max_weight: float = 1.5
    push_to_hub: bool = True


@dataclass(frozen=True)
class ModelVerdict:
    status: str
    accepted: bool
    anomaly: bool
    reason: str


def _compute_batch_loss(model, tokenizer, batch_x, batch_stamp, eval_horizon=None):
    with torch.no_grad():
        s1_tokens, s2_tokens = tokenizer.encode(batch_x, half=True)
    s1_in, s2_in = s1_tokens[:, :-1], s2_tokens[:, :-1]
    s1_tgt, s2_tgt = s1_tokens[:, 1:], s2_tokens[:, 1:]
    stamp_in = batch_stamp[:, :-1, :]
    use_tf = not model.training
    s1_logits, s2_logits = model(
        s1_in, s2_in, stamp=stamp_in, use_teacher_forcing=use_tf, s1_targets=s1_tgt if use_tf else None
    )
    if eval_horizon is not None:
        start_idx, horizon_len = eval_horizon
        end_idx = start_idx + horizon_len
        s1_logits, s2_logits = s1_logits[:, start_idx:end_idx, :], s2_logits[:, start_idx:end_idx, :]
        s1_tgt, s2_tgt = s1_tgt[:, start_idx:end_idx], s2_tgt[:, start_idx:end_idx]
    loss, _, _ = model.head.compute_loss(s1_logits, s2_logits, s1_tgt, s2_tgt)
    return loss


def evaluate_validation_loss(model, tokenizer, val_loader, device, ft_cfg=None):
    model.eval()
    horizon = (max(ft_cfg.context_len - 1, 0), ft_cfg.pred_len) if ft_cfg is not None else None
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for bx, bs in val_loader:
            loss = _compute_batch_loss(model, tokenizer, bx.to(device), bs.to(device), eval_horizon=horizon)
            total_loss += float(loss.item()) * bx.size(0)
            count += bx.size(0)
    return total_loss / max(count, 1)


def _train_one_epoch(model, tokenizer, train_loader, optimizer, scheduler_and_clip):
    scheduler, max_grad_norm = scheduler_and_clip
    device = next(model.parameters()).device
    model.train()
    total_loss, count = 0.0, 0
    for bx, bs in train_loader:
        optimizer.zero_grad()
        loss = _compute_batch_loss(model, tokenizer, bx.to(device), bs.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.item()) * bx.size(0)
        count += bx.size(0)
    return total_loss / max(count, 1)


def _release_accelerator_cache():
    if hasattr(torch, "cuda") and torch.cuda.is_available() and hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()


def _load_frozen_tokenizer(hf_token, device):
    from model import KronosTokenizer

    tokenizer = load_tokenizer_instance(KronosTokenizer, hf_token).to(device)
    for param in tokenizer.parameters():
        param.requires_grad = False
    return tokenizer


def _load_kronos_from_candidates(candidates, device):
    from model import Kronos

    for cand_repo, cand_rev, cand_tok in candidates:
        try:
            kwargs = {"token": cand_tok}
            if cand_rev:
                kwargs["revision"] = cand_rev
            return Kronos.from_pretrained(cand_repo, **kwargs).to(device)
        except Exception:
            continue
    raise RuntimeError(f"Could not load Kronos model from {[c[0] for c in candidates]}")


def _evaluate_champion_baseline(repo_id, branch, hf_token, eval_ctx, ft_cfg):
    tokenizer, val_loader, device = eval_ctx
    fallback_repo = getattr(config, "KRONOS_FALLBACK_MODEL_NAME", "NeoQuasar/Kronos-base")
    candidates = ((repo_id, branch, hf_token), (repo_id, "main", hf_token), (fallback_repo, None, False))
    champion = _load_kronos_from_candidates(candidates, device)
    try:
        return evaluate_validation_loss(champion, tokenizer, val_loader, device, ft_cfg=ft_cfg)
    finally:
        del champion
        _release_accelerator_cache()


def _load_clean_challenger(repo_id, hf_token, device):
    fallback_repo = getattr(config, "KRONOS_FALLBACK_MODEL_NAME", "NeoQuasar/Kronos-base")
    candidates = ((repo_id, "main", hf_token), (fallback_repo, None, False))
    return _load_kronos_from_candidates(candidates, device)


def _build_dataloaders(train_pair, val_pair, ft_cfg):
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    train_ds = TensorDataset(torch.from_numpy(train_pair[0]), torch.from_numpy(train_pair[1]))
    val_ds = TensorDataset(torch.from_numpy(val_pair[0]), torch.from_numpy(val_pair[1]))
    weights = compute_recency_weights(len(train_pair[0]), ft_cfg.recency_min_weight, ft_cfg.recency_max_weight)
    sampler = WeightedRandomSampler(weights=weights.tolist(), num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=ft_cfg.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=ft_cfg.batch_size, shuffle=False)
    return train_loader, val_loader


def _fit_challenger_weights(model, tokenizer, train_loader, val_loader, ft_cfg):
    device = next(model.parameters()).device
    best_new_val = float("inf")
    best_state = None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=ft_cfg.learning_rate, betas=ft_cfg.adam_betas, weight_decay=ft_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=ft_cfg.learning_rate,
        steps_per_epoch=max(len(train_loader), 1),
        epochs=max(ft_cfg.epochs, 1),
        pct_start=0.03,
        div_factor=10.0,
    )
    for epoch in range(1, ft_cfg.epochs + 1):
        train_loss = _train_one_epoch(model, tokenizer, train_loader, optimizer, (scheduler, ft_cfg.max_grad_norm))
        val_loss = evaluate_validation_loss(model, tokenizer, val_loader, device, ft_cfg=ft_cfg)
        print(f"[{ft_cfg.symbol} | Epoch {epoch}/{ft_cfg.epochs}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        if math.isfinite(val_loss) and val_loss < best_new_val:
            best_new_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return best_new_val


def judge_model_update(baseline_val, best_new_val, ft_cfg):
    if not (math.isfinite(baseline_val) and math.isfinite(best_new_val)):
        return ModelVerdict("ANOMALY", False, True, "val_loss is NaN or Inf")
    if best_new_val > ft_cfg.anomaly_val_loss:
        reason = f"val_loss {best_new_val:.4f} > ceiling {ft_cfg.anomaly_val_loss:.2f}"
        return ModelVerdict("ANOMALY", False, True, reason)
    if (best_new_val - baseline_val) > ft_cfg.anomaly_spike_delta:
        reason = f"val_loss spiked +{best_new_val - baseline_val:.4f} vs baseline {baseline_val:.4f}"
        return ModelVerdict("ANOMALY", False, True, reason)
    if best_new_val <= baseline_val + ft_cfg.val_tolerance:
        reason = f"challenger {best_new_val:.4f} <= baseline {baseline_val:.4f} + {ft_cfg.val_tolerance:.3f}"
        return ModelVerdict("UPDATED", True, False, reason)
    reason = f"challenger {best_new_val:.4f} > baseline {baseline_val:.4f} + {ft_cfg.val_tolerance:.3f}"
    return ModelVerdict("KEPT_OLD", False, False, reason)


def _upload_finetuned_branch(model, repo_id, branch, hf_token, metrics):
    if not branch or branch == "main":
        raise ValueError("Refusing to overwrite clean 'main' branch with specialist weights")
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.create_branch(repo_id=repo_id, repo_type="model", branch=branch, revision="main", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        api.upload_folder(
            folder_path=tmp_dir,
            repo_id=repo_id,
            repo_type="model",
            revision=branch,
            commit_message=(
                f"Monthly fine-tune {branch}: val_loss {metrics['baseline_val_loss']:.4f} -> {metrics['best_val_loss']:.4f}"
            ),
        )


def format_finetune_status_vn(metrics):
    status = str((metrics or {}).get("status") or "")
    base_v = (metrics or {}).get("baseline_val_loss")
    best_v = (metrics or {}).get("best_val_loss")
    has_nums = (
        isinstance(base_v, (int, float))
        and isinstance(best_v, (int, float))
        and math.isfinite(base_v)
        and math.isfinite(best_v)
    )
    if status == "UPDATED" and (metrics or {}).get("pushed") and has_nums:
        return f"<code>{base_v:.4f} → {best_v:.4f}</code> (✅ ĐÃ CẬP NHẬT)"
    if status == "ANOMALY":
        if has_nums:
            return f"<code>cũ {base_v:.4f} | mới {best_v:.4f}</code> (⚠️ CẢNH BÁO BẤT THƯỜNG — GIỮ BẢN CŨ)"
        note = html.escape(str((metrics or {}).get("reason") or "lỗi dữ liệu")[:240])
        return f"⚠️ CẢNH BÁO DỮ LIỆU — GIỮ BẢN CŨ (<i>{note}</i>)"
    if has_nums:
        return f"<code>cũ {base_v:.4f} | mới {best_v:.4f}</code> (🛡️ GIỮ BẢN CŨ)"
    return "🛡️ GIỮ BẢN CŨ"


def _build_metrics_summary(ft_cfg, device, baseline_val, best_new_val, verdict):
    finite_both = math.isfinite(baseline_val) and math.isfinite(best_new_val)
    return {
        "symbol": ft_cfg.symbol,
        "branch": ft_cfg.branch,
        "device": device,
        "baseline_val_loss": round(baseline_val, 5) if math.isfinite(baseline_val) else float("inf"),
        "best_val_loss": round(best_new_val, 5) if math.isfinite(best_new_val) else float("inf"),
        "val_delta": round(best_new_val - baseline_val, 5) if finite_both else float("inf"),
        "status": verdict.status,
        "anomaly": verdict.anomaly,
        "reason": verdict.reason,
        "improved": verdict.accepted,
        "pushed": False,
    }


def _log_finetune_outcome(repo_id, ft_cfg, baseline_val, best_new_val, metrics):
    if metrics["pushed"]:
        print(f"[PUSH THÀNH CÔNG] Đã cập nhật {repo_id}@{ft_cfg.branch} (cũ: {baseline_val:.4f} -> mới: {best_new_val:.4f})")
    elif metrics["anomaly"]:
        print(f"[CẢNH BÁO BẤT THƯỜNG] {ft_cfg.symbol}: {metrics['reason']} -> Giữ nguyên bản cũ.")
    else:
        print(
            f"[GIỮ BẢN CŨ] {ft_cfg.symbol}: val_loss mới ({best_new_val:.4f}) > "
            f"bản hiện tại ({baseline_val:.4f}) + dung sai ({ft_cfg.val_tolerance:.3f}), không ghi đè."
        )


def run_specialist_finetune(train_pair, val_pair, ft_cfg: FineTuneConfig, repo_id=None, token=None):
    hf_token = (token or resolve_hf_token() or os.getenv("HF_TOKEN") or "").strip()
    if ft_cfg.push_to_hub and not hf_token:
        raise ValueError("HF_TOKEN with Write access is required to push fine-tuned weights.")

    repo_id = repo_id or getattr(config, "KRONOS_MODEL_NAME", "kingescaped/lvq-std-BOT")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = _load_frozen_tokenizer(hf_token or None, device)
    train_loader, val_loader = _build_dataloaders(train_pair, val_pair, ft_cfg)

    baseline_val = _evaluate_champion_baseline(repo_id, ft_cfg.branch, hf_token or None, (tokenizer, val_loader, device), ft_cfg)
    print(f"[{ft_cfg.symbol}] Điểm chuẩn model tháng trước trên đề thi 30 ngày mới: baseline_val_loss={baseline_val:.4f}")

    challenger = _load_clean_challenger(repo_id, hf_token or None, device)
    best_new_val = _fit_challenger_weights(challenger, tokenizer, train_loader, val_loader, ft_cfg)
    verdict = judge_model_update(baseline_val, best_new_val, ft_cfg)
    metrics = _build_metrics_summary(ft_cfg, device, baseline_val, best_new_val, verdict)

    if ft_cfg.push_to_hub and verdict.accepted:
        _upload_finetuned_branch(challenger, repo_id, ft_cfg.branch, hf_token, metrics)
        metrics["pushed"] = True
    if ft_cfg.push_to_hub:
        _log_finetune_outcome(repo_id, ft_cfg, baseline_val, best_new_val, metrics)
    return metrics
