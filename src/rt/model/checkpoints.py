import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from rt.data import resolve_repo

try:
    _RT_VERSION = version("relational-transformer")
except PackageNotFoundError:
    _RT_VERSION = None

_HF_UA = {"library_name": "relational-transformer", "library_version": _RT_VERSION}

CONFIG_FILE = "config.json"
MODEL_FILE = "model.safetensors"
MODEL_DIM_KEYS = ("num_blocks", "d_model", "d_text", "num_heads", "d_ff")


def save_model(state_dict, path, metadata: dict | None = None) -> None:
    meta = {str(k): str(v) for k, v in (metadata or {}).items()}
    save_file(state_dict, str(path), metadata=meta or None)


def _file_datetime_encoding(path):
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        return (checkpoint.metadata() or {}).get("datetime_encoding")


def _datetime_encoding(config_encoding, file_encoding):
    for source, value in (
        ("config", config_encoding),
        ("file metadata", file_encoding),
    ):
        if value is not None and value not in ("scalar", "calendar"):
            raise ValueError(
                f"unknown datetime_encoding {value!r} in checkpoint {source}"
            )
    if (
        config_encoding is not None
        and file_encoding is not None
        and config_encoding != file_encoding
    ):
        raise ValueError(
            "conflicting datetime_encoding in checkpoint config/request "
            f"({config_encoding!r}) and file metadata ({file_encoding!r})"
        )
    return config_encoding or file_encoding or "scalar"


def validate_datetime_state_dict(state_dict, datetime_encoding: str) -> None:
    """Validate stored calendar-head precision before loading or warm-starting.

    A destination FP32 parameter would silently widen a narrowed source tensor.
    Check the source first; scalar checkpoints retain historical dtype acceptance.
    This does not replace strict key/shape validation by ``load_state_dict``.
    """
    _datetime_encoding(datetime_encoding, None)
    if datetime_encoding != "calendar":
        return
    for name in ("dec_dict.datetime.weight", "dec_dict.datetime.bias"):
        value = state_dict.get(name)
        if value is None:
            raise ValueError(f"calendar checkpoint is missing {name}")
        if value.dtype != torch.float32:
            raise ValueError(
                f"calendar checkpoint requires stored FP32 {name}; got {value.dtype}. "
                "Widening a narrowed head cannot restore its original precision."
            )


def load_model(path, *, datetime_encoding: str | None = None):
    """Load source tensors, validating an explicit or file-tagged representation.

    Pass the encoding returned by ``resolve_checkpoint`` for config-only tags.
    Completely unmarked checkpoints remain scalar; shapes never infer encoding.
    """
    encoding = _datetime_encoding(datetime_encoding, _file_datetime_encoding(path))
    sd = load_file(str(path))
    for k in [k for k in sd if ".boolean" in k]:
        del sd[k]
    validate_datetime_state_dict(sd, encoding)
    return sd


def _compat(config: dict) -> None:
    if "embedder" not in config and "embedding_model" in config:
        config["embedder"] = config.pop("embedding_model")


def resolve_checkpoint(spec, *, revision: str | None = None) -> tuple[dict, Path]:
    p = Path(spec).expanduser()
    if p.is_file():
        cfg_path = p.with_name(CONFIG_FILE)
        config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        model_path = p
    else:
        if p.is_dir():
            d = p
        else:
            repo_id, subdir = resolve_repo(spec)
            local = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=[f"{subdir}/*"] if subdir else None,
                **_HF_UA,
            )
            d = Path(local) / subdir if subdir else Path(local)
        config = json.loads((d / CONFIG_FILE).read_text())
        model_path = d / config.get("checkpoint_file", MODEL_FILE)
    _compat(config)
    file_encoding = _file_datetime_encoding(model_path)
    encoding = _datetime_encoding(
        config.get("model", {}).get("datetime_encoding"), file_encoding
    )
    if file_encoding is not None:
        config.setdefault("model", {})["datetime_encoding"] = encoding
    return config, model_path
