import copy
from dataclasses import dataclass
import os
import tempfile
import torch
import config
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
    push_to_hub: bool = True


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


def _load_starting_models(repo_id, branch, hf_token, device):
    from model import Kronos, KronosTokenizer

    fallback_repo = getattr(config, "KRONOS_FALLBACK_MODEL_NAME", "NeoQuasar/Kronos-base")
    tokenizer = load_tokenizer_instance(KronosTokenizer, hf_token).to(device)
    for param in tokenizer.parameters():
        param.requires_grad = False

    for cand_repo, cand_rev, cand_tok in ((repo_id, branch, hf_token), (repo_id, "main", hf_token), (fallback_repo, None, False)):
        try:
            kwargs = {"token": cand_tok}
            if cand_rev:
                kwargs["revision"] = cand_rev
            model = Kronos.from_pretrained(cand_repo, **kwargs).to(device)
            return model, tokenizer
        except Exception:
            continue
    raise RuntimeError(f"Could not load starting Kronos model from {repo_id} or {fallback_repo}")


def _upload_finetuned_branch(model, repo_id, branch, hf_token, metrics):
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


def _fit_best_weights(model, tokenizer, train_loader, val_loader, ft_cfg):
    device = next(model.parameters()).device
    baseline_val = evaluate_validation_loss(model, tokenizer, val_loader, device, ft_cfg=ft_cfg)
    best_val = baseline_val
    best_state = copy.deepcopy(model.state_dict())
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
        if val_loss <= best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return baseline_val, best_val


def run_specialist_finetune(train_pair, val_pair, ft_cfg: FineTuneConfig, repo_id=None, token=None):
    hf_token = (token or resolve_hf_token() or os.getenv("HF_TOKEN") or "").strip()
    if ft_cfg.push_to_hub and not hf_token:
        raise ValueError("HF_TOKEN with Write access is required to push fine-tuned weights.")

    from torch.utils.data import DataLoader, TensorDataset

    repo_id = repo_id or getattr(config, "KRONOS_MODEL_NAME", "kingescaped/lvq-std-BOT")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = _load_starting_models(repo_id, ft_cfg.branch, hf_token or None, device)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_pair[0]), torch.from_numpy(train_pair[1])), batch_size=ft_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_pair[0]), torch.from_numpy(val_pair[1])), batch_size=ft_cfg.batch_size, shuffle=False)

    baseline_val, best_val = _fit_best_weights(model, tokenizer, train_loader, val_loader, ft_cfg)
    improved = best_val <= baseline_val + 1e-6
    metrics = {
        "symbol": ft_cfg.symbol, "branch": ft_cfg.branch, "device": device,
        "baseline_val_loss": round(baseline_val, 5), "best_val_loss": round(best_val, 5),
        "improved": improved, "pushed": False,
    }
    if ft_cfg.push_to_hub and improved:
        _upload_finetuned_branch(model, repo_id, ft_cfg.branch, hf_token, metrics)
        metrics["pushed"] = True
        print(f"[PUSH THÀNH CÔNG] Đã cập nhật {repo_id}@{ft_cfg.branch} (val_loss: {baseline_val:.4f} -> {best_val:.4f})")
    elif ft_cfg.push_to_hub:
        print(f"[GIỮ BẢN CŨ] {ft_cfg.symbol}: val_loss mới ({best_val:.4f}) > bản hiện tại ({baseline_val:.4f}), không ghi đè.")
    return metrics
