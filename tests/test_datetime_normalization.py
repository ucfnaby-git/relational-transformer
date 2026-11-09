"""Native CPU regression for retained-cell fitting, not temporal neighbor filtering."""

import json
from decimal import Decimal, localcontext

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from rt.rustler import Sampler, preprocess

from rt.data.datasets import process_batch

NS = 1_000_000_000
CUTOFF = 946684800 * NS  # 2000-01-01 UTC; all fixture offsets fit timestamp[ns].


def source(variant):
    # Each row is (row time, another retained datetime), relative to the cutoff.
    data = {
        "Db": [
            (-1, -3 * NS),
            (0, 5 * NS),
            (1, -100 * NS),
            (86400 * NS, 100 * NS),
            (None, 7 * NS),
        ],
        "Train": [(2 * NS, -2 * NS), (4 * NS, None)],
        "Val": [(-5 * 10**18, 6 * 10**18)],
        "Test": [(6 * 10**18, -5 * 10**18)],
    }
    if variant == "val_test":
        data.update(Val=[(-4 * 10**18, 5 * 10**18)], Test=[(5 * 10**18, -4 * 10**18)])
    elif variant == "db_future":
        data["Db"][2:] = [
            (2, 5 * 10**18),
            (172800 * NS, -4 * 10**18),
            (None, 6 * 10**18),
        ]
    elif variant == "train":
        data["Train"][0] = (20 * NS, -2 * NS)
    elif variant == "after":
        data["Db"][1] = (1, 5 * NS)
    return data


def sample(pre, split, column, count):
    table = "events" if split == "Db" else "forecast"
    db = pre / "scope"
    info = json.loads((db / "table_info.json").read_text())[f"{table}:{split}"]
    columns = json.loads((db / "column_index.json").read_text())
    sampler = Sampler(
        dataset_tuples=[("scope", table, info["node_idx_offset"], count)],
        global_rank=0,
        local_rank=0,
        world_size=1,
        local_ctx_size_list=[8],
        bfs_width_list=[4],
        num_walks=0,
        walk_length=0,
        prefer_latest_list=[False],
        mask_prob_max=0.0,
        embedder="test",
        pre_dir=str(pre),
        d_text=1,
        shuffle_seed=0,
        context_seed=0,
        target_columns=[columns[f"{column} of {table}"]],
        columns_to_drop=[[]],
        cutoff_timestamps=[None],
        items_per_task=-1,
        quiet=True,
        ignore_data_errors=False,
        num_prev_skipped=0,
        mmap_populate=False,
        legacy_boolean=False,
        timeout_per_item=1.0,
        vector_db_path=None,
        exact_datetime=True,
    )
    # An integer index bounds sampling even when a primary date is null.
    batch = process_batch(sampler.batch_py(0, count, 8), d_text=1)
    assert batch["datetime_features"].dtype == torch.bfloat16
    assert batch["datetime_target_values"].dtype == torch.float32
    return batch, info["node_idx_offset"]


def observe(path, variant):
    data = source(variant)
    raw, pre = path / "raw", path / "pre"
    (raw / "db").mkdir(parents=True)
    (raw / "tasks/forecast").mkdir(parents=True)
    (raw / "manifest.yaml").write_text(
        "name: scope\nval_timestamp: '2000-01-01'\ntables:\n  events:\n    time_col: time\n"
    )
    (raw / "tasks/forecast/manifest.yaml").write_text(
        "target_col: value\ntask_type: regression\ntime_col: time\n"
    )
    for split, rows in data.items():
        columns = {
            name: pa.array(
                [None if row[i] is None else CUTOFF + row[i] for row in rows],
                type=pa.timestamp("ns"),
            )
            for i, name in enumerate(("time", "extra"))
        }
        columns["value"] = pa.array(range(len(rows)), type=pa.float64())
        filename = (
            "db/events.parquet"
            if split == "Db"
            else f"tasks/forecast/{split.lower()}.parquet"
        )
        pq.write_table(pa.table(columns), raw / filename)
    preprocess(str(raw), str(pre), skip_tasks=False, exact_datetime=True)
    db = pre / "scope"
    texts = json.loads((db / "text.json").read_text())
    np.zeros(len(texts), dtype=np.uint16).tofile(db / "text_emb_test.bin")
    fit = json.loads((db / "datetime_exact.json").read_text())["normalization"]

    # Explicit source-row eligibility, independent of native metadata or masks.
    eligible = data["Db"][: 1 if variant == "after" else 2] + data["Train"]
    with localcontext() as ctx:
        ctx.prec = 80
        population = [Decimal(x) / NS for row in eligible for x in row if x is not None]
        mean = sum(population) / len(population)
        std = (sum((x - mean) ** 2 for x in population) / len(population)).sqrt()
        actual_mean = (
            Decimal(fit["origin_seconds"] - CUTOFF // NS)
            + Decimal(fit["origin_nanoseconds"]) / NS
            + Decimal(str(fit["mean_offset_seconds"]))
        )
        assert fit["count"] == len(population)
        assert abs(actual_mean - mean) < Decimal("1e-14")
        assert abs(Decimal(str(fit["std_seconds"])) - std) < Decimal("1e-14")
        cells = {}
        for split, rows in data.items():
            for col, name in enumerate(("time", "extra")):
                batch, offset = sample(pre, split, name, len(rows))
                observed = set()
                for i in batch["batch_mask"].nonzero().flatten().tolist():
                    row = int(batch["node_idxs"][i, 0]) - offset
                    value = rows[row][col]
                    assert value is not None
                    assert batch["datetime_valid"][i, 0] and batch["is_targets"][i, 0]
                    assert (
                        int(batch["datetime_seconds"][i, 0]),
                        int(batch["datetime_nanoseconds"][i, 0]),
                    ) == divmod(CUTOFF + value, NS)
                    target = float(batch["datetime_target_values"][i, 0, 0])
                    expected = np.float32(float((Decimal(value) / NS - mean) / std))
                    # Allow one final FP32 rounding ULP; do not require ns-distinct scalars.
                    np.testing.assert_array_max_ulp(
                        np.float32(target), expected, maxulp=1
                    )
                    cells[split, row, col] = (
                        tuple(batch["datetime_features"][i, 0].tolist()),
                        target,
                    )
                    observed.add(row)
                assert observed == {
                    i for i, row in enumerate(rows) if row[col] is not None
                }
    return fit, cells


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    return observe(tmp_path_factory.mktemp("datetime-normalization"), "baseline")


def test_inclusive_cutoff_null_row_time_and_future_train_targets(baseline):
    fit, cells = baseline
    assert fit["count"] == 7  # Four Db cells plus three future/other Train cells.
    assert len({cells["Db", row, 0][0] for row in range(3)}) == 3
    assert ("Db", 4, 0) not in cells and ("Db", 4, 1) in cells
    assert ("Train", 0, 0) in cells and ("Train", 1, 0) in cells


@pytest.mark.parametrize("variant", ["val_test", "db_future"])
def test_excluded_values_change_features_without_refitting(tmp_path, baseline, variant):
    original_fit, original = baseline
    fit, cells = observe(tmp_path, variant)
    assert fit == original_fit
    changed = {key for key in cells if cells[key][0] != original[key][0]}
    if variant == "val_test":
        expected = {key for key in cells if key[0] in ("Val", "Test")}
    else:
        expected = {key for key in cells if key[0] == "Db" and key[1] >= 2}
    assert changed == expected and changed
    assert all(cells[key][1] == original[key][1] for key in cells if key not in changed)


@pytest.mark.parametrize("variant,count", [("train", 7), ("after", 5)])
def test_train_change_and_one_ns_cutoff_crossing_refit(
    tmp_path, baseline, variant, count
):
    fit, _ = observe(tmp_path, variant)
    assert fit["count"] == count
    assert fit != baseline[0]
