"""Opt-in TorchNPU automatic migration for the RT-J entry points."""

import os

AUTO_MIGRATE_NPU = os.environ.get("RT_AUTO_MIGRATE_NPU") == "1"

if AUTO_MIGRATE_NPU:
    # PyTorch broadcast queries CUDA capability even for non-FP8 tensors.
    # This compatibility value disables CUDA SM-specific paths on NPU.
    os.environ.setdefault("TORCH_NPU_DEVICE_CAPABILITY", "0.0")
    import torch  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
