import numpy as np
import torch

from rt.data.datasets import process_batch


def bf16_storage(values):
    return (
        torch.tensor(values, dtype=torch.float32).bfloat16().view(torch.int16).numpy()
    )


def test_exact_datetime_batch_keeps_integer_source_and_fp32_targets():
    # These normalized values collide in the original BF16 scalar path.
    targets = np.array([1.0196260213851929, 1.0262432098388672], dtype=np.float32)
    assert torch.from_numpy(targets).bfloat16().unique().numel() == 1
    values = {
        name: np.zeros(2, dtype=np.int32)
        for name in (
            "node_idxs",
            "sem_types",
            "table_name_idxs",
            "col_name_idxs",
            "class_value_idxs",
            "timestamps",
            "seed_node_idxs",
            "bfs_depths",
        )
    }
    values.update(
        {
            name: np.zeros(2, dtype=bool)
            for name in ("is_targets", "is_task_nodes", "is_padding")
        }
    )
    features = np.tile(
        np.array(
            [0, 1, 255, 7, 12, 31, 23, 59, 3, 231, 3, 231, 3, 231], dtype=np.float32
        )
        / 256,
        2,
    )
    values.update(
        seq_len=2,
        f2p_nbr_idxs=np.full(10, -1, dtype=np.int32),
        number_values=bf16_storage([0, 0]),
        datetime_values=bf16_storage(targets),
        text_values=bf16_storage([0, 0, 0, 0]),
        col_name_values=bf16_storage([0, 0, 0, 0]),
        datetime_features=bf16_storage(features),
        datetime_target_values=targets,
        # A negative subsecond and year 10000; neither goes through epoch-ns FP32.
        datetime_seconds=np.array([-1, 253402325940], dtype=np.int64),
        datetime_nanoseconds=np.array([999999999, 1], dtype=np.int32),
        datetime_valid=np.ones(2, dtype=bool),
    )
    batch = process_batch(values.items(), d_text=2)
    assert batch["datetime_seconds"].tolist() == [[-1, 253402325940]]
    assert batch["datetime_seconds"].dtype == torch.int64
    assert batch["datetime_nanoseconds"].tolist() == [[999999999, 1]]
    assert batch["datetime_nanoseconds"].dtype == torch.int32
    assert batch["datetime_valid"].dtype == torch.bool
    assert batch["datetime_features"].shape == (1, 2, 14)
    assert batch["datetime_features"].dtype == torch.bfloat16
    torch.testing.assert_close(
        batch["datetime_features"].float().flatten(),
        torch.from_numpy(features),
        atol=0,
        rtol=0,
    )
    assert batch["datetime_target_values"].shape == (1, 2, 1)
    assert batch["datetime_target_values"].dtype == torch.float32
    torch.testing.assert_close(
        batch["datetime_target_values"].flatten(),
        torch.from_numpy(targets),
        atol=0,
        rtol=0,
    )
    assert batch["datetime_target_values"].unique().numel() == 2
