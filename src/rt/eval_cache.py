"""Datetime representation versions for evaluation outputs, not weight identity."""

DATETIME_CACHE_KEY = "datetime_context_version"
CALENDAR_DATETIME_CACHE_VERSION = "calendar_datetime_v2"


def datetime_cache_version(encoding):
    if encoding not in ("scalar", "calendar"):
        raise ValueError(f"unknown datetime_encoding {encoding!r}")
    return CALENDAR_DATETIME_CACHE_VERSION if encoding == "calendar" else "scalar"


def check_datetime_cache(state, expected):
    # Legacy scalar JSON/NPZ files have no marker. Never infer calendar19 from
    # their existence or the still-unchanged datetime_encoding="calendar" flag.
    saved = str(state.get(DATETIME_CACHE_KEY, "scalar"))
    if saved != expected:
        raise ValueError(
            "Saved evaluation results use a different datetime context version "
            f"(saved={saved!r}, expected={expected!r}); "
            "use a fresh run_id/output directory"
        )


def checkpoint_datetime_cache_version(ckpt):
    """Resolve lazily for cache hits without constructing a model or using CUDA.

    Reuse the existing config/file metadata conflict checks. Calendar header
    shape validation prevents an incompatible or unmarked checkpoint from
    bypassing its rejection through an existing final result. No payloads or hashes are
    needed here; normal model loading retains the complete validation contract.
    """
    from safetensors import safe_open

    from rt.model.checkpoints import resolve_checkpoint

    config, path = resolve_checkpoint(ckpt)
    encoding = config.get("model", {}).get("datetime_encoding", "scalar")
    version = datetime_cache_version(encoding)
    with safe_open(str(path), framework="np") as checkpoint:
        name = "enc_dict.datetime.weight"
        names = checkpoint.keys()
        if name not in names:
            raise ValueError(f"{encoding} checkpoint is missing {name}")
        shape = checkpoint.get_slice(name).get_shape()
    width = 19 if encoding == "calendar" else 1
    if len(shape) != 2 or shape[1] != width:
        raise ValueError(
            f"{encoding} evaluation requires a {width}-coordinate datetime adapter; "
            f"checkpoint shape is {shape}. Use a compatible checkpoint"
        )
    return version
