import os
import config

_SHARED_TOKENIZER = None
_PREDICTOR_CACHE = {}


def resolve_hf_token():
    raw = getattr(config, "HF_TOKEN", None) or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    token = str(raw).strip() if raw else ""
    if not token:
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        return None
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token


def reset_loader_cache():
    global _SHARED_TOKENIZER
    _SHARED_TOKENIZER = None
    _PREDICTOR_CACHE.clear()


def load_tokenizer_instance(tokenizer_cls, hf_token):
    model_repo = getattr(config, "KRONOS_MODEL_NAME", "kingescaped/lvq-std-BOT")
    tok_repo = getattr(config, "KRONOS_TOKENIZER_NAME", model_repo)
    raw_rev = getattr(config, "KRONOS_TOKENIZER_REVISION", "tokenizer")
    eff_rev = raw_rev if (tok_repo == model_repo or os.getenv("KRONOS_TOKENIZER_REVISION")) else ""
    fallback_tok = getattr(config, "KRONOS_FALLBACK_TOKENIZER_NAME", "NeoQuasar/Kronos-Tokenizer-base")

    candidates = [(tok_repo, eff_rev or None, hf_token)]
    if eff_rev and tok_repo != model_repo:
        candidates.append((tok_repo, None, hf_token))
    if (fallback_tok, None) != (tok_repo, eff_rev or None):
        candidates.append((fallback_tok, None, False))

    last_err = None
    for repo_id, rev, tok in candidates:
        label = f"{repo_id}@{rev}" if rev else repo_id
        try:
            print(f"[AI KRONOS] Tải Tokenizer: {label}")
            kwargs = {"token": tok}
            if rev:
                kwargs["revision"] = rev
            instance = tokenizer_cls.from_pretrained(repo_id, **kwargs)
            instance.eval()
            return instance
        except Exception as err:
            last_err = err
            print(f"[CẢNH BÁO AI] Chưa nạp được Tokenizer {label} ({err}) -> thử nguồn dự phòng...")
    raise RuntimeError(last_err or "Could not load KronosTokenizer")


def _load_shared_tokenizer(tokenizer_cls, hf_token):
    global _SHARED_TOKENIZER
    if _SHARED_TOKENIZER is not None:
        return _SHARED_TOKENIZER
    _SHARED_TOKENIZER = load_tokenizer_instance(tokenizer_cls, hf_token)
    return _SHARED_TOKENIZER


def _build_model_candidates(primary_repo, revision, fallback_repo, hf_token):
    candidates = []
    seen = set()

    def add_candidate(repo_id, rev, tok):
        if not repo_id:
            return
        key = (repo_id, rev or "")
        if key not in seen:
            seen.add(key)
            candidates.append((repo_id, rev, tok))

    add_candidate(primary_repo, revision, hf_token)
    if revision and revision != "main":
        add_candidate(primary_repo, "main", hf_token)
    add_candidate(fallback_repo, None, False)
    return candidates


def _instantiate_from_candidate(kronos_cls, predictor_cls, tokenizer, candidate):
    repo_id, rev, tok = candidate
    source_label = f"{repo_id}@{rev}" if rev else repo_id
    print(f"[AI KRONOS] Tải Mô hình: {source_label}")
    kwargs = {"token": tok}
    if rev:
        kwargs["revision"] = rev
    model = kronos_cls.from_pretrained(repo_id, **kwargs)
    model.eval()
    predictor = predictor_cls(model=model, tokenizer=tokenizer, device="cpu", max_context=512)
    try:
        predictor.model_source = source_label
    except AttributeError:
        pass
    print(f"[AI KRONOS] Đã nạp thành công {source_label} vào bộ nhớ!\n")
    return predictor


def load_predictor_for_target(symbol_name=None, revision=None):
    primary_repo = getattr(config, "KRONOS_MODEL_NAME", "kingescaped/lvq-std-BOT")
    fallback_repo = getattr(config, "KRONOS_FALLBACK_MODEL_NAME", "NeoQuasar/Kronos-base")
    cache_key = (primary_repo, revision or "")
    if cache_key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[cache_key]

    try:
        target_desc = f"{symbol_name} ({revision})" if symbol_name and revision else (primary_repo)
        print(f"\n[AI KRONOS] Đang khởi động mô hình Kronos cho {target_desc}...")
        from model import Kronos, KronosPredictor, KronosTokenizer

        hf_token = resolve_hf_token()
        tokenizer = _load_shared_tokenizer(KronosTokenizer, hf_token)
        candidates = _build_model_candidates(primary_repo, revision, fallback_repo, hf_token)

        last_err = None
        for candidate in candidates:
            try:
                predictor = _instantiate_from_candidate(Kronos, KronosPredictor, tokenizer, candidate)
                _PREDICTOR_CACHE[cache_key] = predictor
                return predictor
            except Exception as err:
                last_err = err
                src = f"{candidate[0]}@{candidate[1]}" if candidate[1] else candidate[0]
                print(f"[CẢNH BÁO AI] Chưa nạp được {src} ({err}) -> thử nguồn dự phòng tiếp theo...")

        raise RuntimeError(last_err or "No valid Kronos model candidate")
    except Exception as exc:
        print(f"[CẢNH BÁO AI] Không nạp được Kronos ({exc}). Tạm dừng bắn tín hiệu tự động.")
        return None
