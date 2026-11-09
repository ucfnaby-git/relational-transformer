import json
from types import SimpleNamespace

from rt.data import datasets


def test_skipped_cutoff_does_not_shift_surviving_task_labels(tmp_path, monkeypatch):
    tasks = []
    for db in ("broken", "valid"):
        path = tmp_path / db
        path.mkdir()
        (path / "table_info.json").write_text(
            json.dumps({"events:Train": {"node_idx_offset": 0, "num_nodes": 2}})
        )
        tasks.append(
            SimpleNamespace(
                db_name=db,
                table_name="events",
                target_column="label",
                split="train",
                leakage_columns=[("events", "leak")],
            )
        )
    monkeypatch.setattr(datasets, "resolve_pre_dir", lambda path: path)
    monkeypatch.setattr(
        datasets,
        "get_column_index",
        lambda column, table, db, pre_dir: (
            (10 if db == "broken" else 20) + (1 if column == "label" else 2)
        ),
    )

    def cutoff(option, db, pre_dir):
        if db == "broken":
            raise ValueError("missing cutoff")
        return 123

    monkeypatch.setattr(datasets, "_resolve_cutoff", cutoff)
    captured = {}

    def sampler(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(num_items=2)

    monkeypatch.setattr(datasets, "Sampler", sampler)
    datasets.RustlerDataset(
        tasks=tasks,
        pre_dir=str(tmp_path),
        global_rank=0,
        local_rank=0,
        world_size=1,
        local_ctx_size_list=[4],
        bfs_width_list=[1],
        num_walks=1,
        walk_length=1,
        prefer_latest_list=[True],
        mask_prob_max=0,
        embedder="unused",
        d_text=2,
        shuffle_seed=0,
        context_seed=0,
        items_per_task=2,
        quiet=True,
        ignore_data_errors=True,
        mmap_populate=False,
        legacy_boolean=False,
        timeout_per_item=1,
        vector_db_path=None,
        db_cutoff="val",
    )
    assert captured["dataset_tuples"] == [("valid", "events", 0, 2)]
    assert captured["target_columns"] == [21]
    assert captured["columns_to_drop"] == [[22]]
    assert captured["cutoff_timestamps"] == [123]
    assert captured["num_prev_skipped"] == 1
