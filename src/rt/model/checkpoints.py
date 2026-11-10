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


def _file_encoding_metadata(path):
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        return checkpoint.metadata() or {}


_ENCODING_MODES = {
    "datetime_encoding": ("scalar", "calendar"),
    "numeric_encoding": ("scalar", "bits"),
}


def _encoding(name, config_encoding, file_encoding):
    for source, value in (
        ("config", config_encoding),
        ("file metadata", file_encoding),
    ):
        if value is not None and value not in _ENCODING_MODES[name]:
            raise ValueError(f"unknown {name} {value!r} in checkpoint {source}")
    if (
        config_encoding is not None
        and file_encoding is not None
        and config_encoding != file_encoding
    ):
        raise ValueError(
            f"conflicting {name} in checkpoint config/request "
            f"({config_encoding!r}) and file metadata ({file_encoding!r})"
        )
    return config_encoding or file_encoding or "scalar"


def validate_precision_state_dict(
    state_dict, *, datetime_encoding: str = "scalar", numeric_encoding: str = "scalar"
) -> None:
    """Validate calendar adapter compatibility and protected source heads.

    A destination FP32 parameter silently widens narrowed source tensors.
    Validate source tensors first; scalar heads retain historical dtype acceptance.
    This does not replace strict key/shape validation by load_state_dict.
    """
    for name, mode, head in (
        ("datetime_encoding", datetime_encoding, "datetime"),
        ("numeric_encoding", numeric_encoding, "number"),
    ):
        _encoding(name, mode, None)
        if mode == "scalar":
            continue
        if head == "datetime":
            adapter = state_dict.get("enc_dict.datetime.weight")
            if adapter is not None and adapter.ndim == 2 and adapter.shape[1] == 14:
                raise ValueError(
                    "calendar checkpoint has an obsolete 14-coordinate datetime adapter; "
                    "this model requires 19 coordinates (calendar14 + normalized FP32 "
                    "cue/bytes). Use a new initialization or explicit checkpoint "
                    "migration with fresh optimizer/SWA state; no automatic conversion."
                )
        for key in (f"dec_dict.{head}.weight", f"dec_dict.{head}.bias"):
            value = state_dict.get(key)
            if value is None:
                raise ValueError(f"{mode} checkpoint is missing {key}")
            if value.dtype != torch.float32:
                raise ValueError(
                    f"{mode} checkpoint requires stored FP32 {key}; got {value.dtype}. "
                    "Widening a narrowed head cannot restore its original precision."
                )


def validate_datetime_state_dict(state_dict, datetime_encoding: str) -> None:
    """Compatibility wrapper for callers validating only the datetime mode."""
    validate_precision_state_dict(state_dict, datetime_encoding=datetime_encoding)


def load_model(
    path, *, datetime_encoding: str | None = None, numeric_encoding: str | None = None
):
    """Load tensors with independent representation and source-head validation.

    Pass both encodings returned by resolve_checkpoint for config-only tags.
    Missing tags remain scalar; tensor shapes never infer a representation.
    """
    metadata = _file_encoding_metadata(path)
    datetime_encoding = _encoding(
        "datetime_encoding", datetime_encoding, metadata.get("datetime_encoding")
    )
    numeric_encoding = _encoding(
        "numeric_encoding", numeric_encoding, metadata.get("numeric_encoding")
    )
    sd = load_file(str(path))
    for k in [k for k in sd if ".boolean" in k]:
        del sd[k]
    validate_precision_state_dict(
        sd, datetime_encoding=datetime_encoding, numeric_encoding=numeric_encoding
    )
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
    metadata = _file_encoding_metadata(model_path)
    for name in _ENCODING_MODES:
        file_encoding = metadata.get(name)
        encoding = _encoding(name, config.get("model", {}).get(name), file_encoding)
        if file_encoding is not None:
            config.setdefault("model", {})[name] = encoding
    return config, model_path
