"""CPU lifecycle checks; no real model, CUDA, dataset or training job needed."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from examples.icl import ensemble, run, tune
from rt.eval import _eval
from rt.eval_cache import (
    CALENDAR_DATETIME_CACHE_VERSION as V2,
)
from rt.eval_cache import (
    DATETIME_CACHE_KEY as KEY,
)
from rt.eval_cache import (
    checkpoint_datetime_cache_version,
)


def checkpoint(
    tmp_path,
    encoding="calendar",
    width=None,
    tag="config",
    *,
    numeric_encoding="scalar",
    numeric_width=None,
    numeric_tag="config",
):
    root = tmp_path / "checkpoint"
    root.mkdir(exist_ok=True)
    config, metadata = {"model": {}}, {}
    if tag == "config":
        config["model"]["datetime_encoding"] = encoding
    elif tag == "file":
        metadata["datetime_encoding"] = encoding
    if numeric_tag == "config":
        config["model"]["numeric_encoding"] = numeric_encoding
    elif numeric_tag == "file":
        metadata["numeric_encoding"] = numeric_encoding
    (root / "config.json").write_text(json.dumps(config))
    width = width if width is not None else (19 if encoding == "calendar" else 1)
    numeric_width = (
        numeric_width
        if numeric_width is not None
        else (5 if numeric_encoding == "bits" else 1)
    )
    save_file(
        {
            "enc_dict.datetime.weight": torch.zeros(2, width),
            "dec_dict.datetime.weight": torch.zeros(1, 2),
            "dec_dict.datetime.bias": torch.zeros(1),
            "enc_dict.number.weight": torch.zeros(2, numeric_width),
        },
        root / "model.safetensors",
        metadata=metadata,
    )
    return str(root)


def worker_args(tmp_path, ckpt):
    return {
        "variant": "test",
        "device": "cpu",
        "db": "db",
        "table": "task",
        "ctx_size": 2,
        "local_ctx_size": 2,
        "bfs_width": 1,
        "prefer_latest": True,
        "n_seeds": 2,
        "items_per_task": 2,
        "split": "test",
        "pre_dir": "unused",
        "out_dir": str(tmp_path),
        "num_walks": 0,
        "walk_length": 0,
        "shuffle_seed": 0,
        "context_seed": 0,
        "tokens_per_gpu": 4,
        "num_workers": 0,
        "prefetch_factor": 1,
        "mmap_populate": False,
        "db_cutoff": None,
        "ckpt": ckpt,
    }


def no_model(*args, **kwargs):
    pytest.fail("cache check must precede model construction")


def test_version_helpers_import_without_torch():
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import rt.eval_cache; assert 'torch' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("tag", ["config", "file"])
def test_checkpoint_version_checks_calendar_header_before_cache_hit(tmp_path, tag):
    path = checkpoint(tmp_path, tag=tag)
    assert checkpoint_datetime_cache_version(path) == V2
    assert not torch.cuda.is_initialized()
    checkpoint(tmp_path, width=14, tag=tag)
    (tmp_path / "db__task.json").write_text(json.dumps({"config": {KEY: V2}}))
    with pytest.raises(ValueError, match="19-coordinate"):
        run.main(**worker_args(tmp_path, path))


@pytest.mark.parametrize("width", [1, 14, 19])
def test_unmarked_header_checked_by_worker_and_both_planners(
    tmp_path, monkeypatch, width
):
    ckpt = checkpoint(tmp_path, "scalar", width=width, tag="none")
    monkeypatch.setattr(tune.env, "out_root", lambda: str(tmp_path))
    monkeypatch.setattr(tune.env, "pre_dir", lambda: "unused")
    monkeypatch.setattr(
        ensemble,
        "load_configs",
        lambda *a: {"db/task": {"top_cfgs": [[2, 2, 1, True]]}},
    )
    grid = "unmarked--{db}--{table}"
    tuning = tune.grid_path(grid, "db", "task")
    tuning.parent.mkdir(parents=True)
    tuning.write_text(json.dumps({"db/task": {}}))
    final = tmp_path / "db__task.json"
    final.write_text(json.dumps({"config": {}}))
    planner_final = tmp_path / "results/cfg0/db__task.json"
    planner_final.parent.mkdir(parents=True)
    planner_final.write_bytes(final.read_bytes())
    paths = [tuning, final, planner_final]
    before = [p.read_bytes() for p in paths]
    args = {"ckpt": ckpt, "grid_stem": grid, "task_list": [("db", "task")]}
    calls = [
        lambda: checkpoint_datetime_cache_version(ckpt),
        lambda: run.main(**worker_args(tmp_path, ckpt)),
        lambda: tune.jobs(**args),
        lambda: ensemble.jobs(**args, configs_path="unused", out_subdir="results"),
    ]
    for call in calls:
        if width == 1:
            assert call() in ("scalar", None, [])
        else:
            with pytest.raises(ValueError, match="1-coordinate"):
                call()
    assert [p.read_bytes() for p in paths] == before
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("shape", [None, (2,), (2, 1, 1), (2, 0)])
def test_cache_header_rejects_missing_or_malformed_adapter(tmp_path, shape):
    ckpt = checkpoint(tmp_path, "scalar", tag="none")
    tensors = {"dec_dict.datetime.bias": torch.zeros(1)}
    if shape is not None:
        tensors["enc_dict.datetime.weight"] = torch.zeros(shape)
    save_file(tensors, tmp_path / "checkpoint/model.safetensors")
    with pytest.raises(ValueError, match="missing|1-coordinate"):
        checkpoint_datetime_cache_version(ckpt)


@pytest.mark.parametrize(
    "encoding,marker,accepted",
    [
        ("calendar", None, False),
        ("calendar", "calendar_datetime_v1", False),
        ("calendar", V2, True),
        ("scalar", None, True),
        ("scalar", "scalar", True),
        ("scalar", V2, False),
    ],
)
def test_final_cache_checked_before_model_load(
    tmp_path, monkeypatch, encoding, marker, accepted
):
    import rt.model

    monkeypatch.setattr(rt.model, "load_rt_model", no_model)
    ckpt = checkpoint(
        tmp_path, encoding, tag="none" if encoding == "scalar" else "config"
    )
    final = tmp_path / "db__task.json"
    final.write_text(json.dumps({"config": {} if marker is None else {KEY: marker}}))
    before = final.read_bytes()
    if accepted:
        run.main(**worker_args(tmp_path, ckpt))
    else:
        with pytest.raises(ValueError, match="datetime context version"):
            run.main(**worker_args(tmp_path, ckpt))
    assert final.read_bytes() == before


@pytest.mark.parametrize("marker", [None, "calendar_datetime_v1"])
def test_partial_cache_rejected_before_accumulator_or_model_load(
    tmp_path, monkeypatch, marker
):
    import rt.model

    monkeypatch.setattr(rt.model, "load_rt_model", no_model)
    ckpt = checkpoint(tmp_path)
    path = tmp_path / "db__task.state.npz"
    # No accumulator fields: a version error must occur before reading them,
    # even when the old record says all requested seeds are already complete.
    np.savez(path, seeds=2, **({} if marker is None else {KEY: marker}))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="datetime context version"):
        run.main(**worker_args(tmp_path, ckpt))
    assert path.read_bytes() == before
    assert not (tmp_path / "db__task.json").exists()


@pytest.mark.parametrize("encoding", ["scalar", "calendar"])
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize(
    "numeric_encoding,exact_numeric",
    [("scalar", False), ("scalar", True), ("bits", False)],
)
def test_worker_stamps_outputs_and_resumes_compatible_state(
    tmp_path, monkeypatch, encoding, resume, numeric_encoding, exact_numeric
):
    import rt.data
    import rt.eval
    import rt.eval.metrics
    import rt.model

    ckpt = checkpoint(tmp_path, encoding, numeric_encoding=numeric_encoding)
    truth = exact_numeric or numeric_encoding == "bits"
    version = V2 if encoding == "calendar" else "scalar"
    task = SimpleNamespace(db_name="db", table_name="task", task_type="reg")
    labels, nodes = np.array([1.0001, 3.0002], dtype=np.float32), np.array([10, 11])
    calls = []
    model = SimpleNamespace(
        datetime_encoding=encoding, numeric_encoding=numeric_encoding
    )
    model.set_compute_dtype = lambda dtype: model
    monkeypatch.setattr(rt.data, "get_tasks", lambda *args: [task])
    monkeypatch.setattr(
        rt.model,
        "load_rt_model",
        lambda *a, **k: (model, {"embedder": "test", "d_text": 2}),
    )

    class Evaluator:
        def evaluate_raw(self, *args, **kwargs):
            calls.append(True)
            yield task, 2, labels, {"": np.array([3.0, 5.0])}, None, nodes

    def evaluator(*args, **kwargs):
        assert kwargs["exact_numeric"] == truth
        assert kwargs["exact_datetime"] == (encoding == "calendar")
        return Evaluator()

    monkeypatch.setattr(rt.eval, "build_evaluator", evaluator)
    monkeypatch.setattr(
        rt.eval.metrics,
        "metric_for",
        lambda _t, y, p: ("mae", float(np.abs(y - p).mean())),
    )
    state = tmp_path / "db__task.state.npz"
    if resume:
        # Unmarked scalar state is deliberately retained as a legacy case.
        np.savez(
            state,
            sum_preds=np.array([1.0, 3.0]),
            labels=labels,
            node_idxs=nodes,
            seeds=1,
            curve=json.dumps({"1": 0.0}),
            **({KEY: version} if encoding == "calendar" else {}),
            **(
                {"numeric_encoding": numeric_encoding, "exact_numeric": truth}
                if truth
                else {}
            ),
        )
    args = worker_args(tmp_path, ckpt) | {"exact_numeric": exact_numeric}
    run.main(**args)
    assert len(calls) == (1 if resume else 2)
    with np.load(state) as saved:
        assert str(saved[KEY]) == version
        assert int(saved["seeds"]) == 2
        assert str(saved["numeric_encoding"]) == numeric_encoding
        assert bool(saved["exact_numeric"]) == truth
        assert saved["labels"].dtype == np.float32
        np.testing.assert_array_equal(saved["labels"], labels)
        np.testing.assert_array_equal(saved["sum_preds"], [4, 8] if resume else [6, 10])
    final = json.loads((tmp_path / "db__task.json").read_text())
    assert final["config"][KEY] == version
    assert final["config"]["numeric_encoding"] == numeric_encoding
    assert final["config"]["exact_numeric"] == truth
    assert set(final["curve"]) == {"1", "2"}
    monkeypatch.setattr(rt.model, "load_rt_model", no_model)
    run.main(**args)
    assert len(calls) == (1 if resume else 2)


@pytest.mark.parametrize("encoding", ["scalar", "calendar"])
@pytest.mark.parametrize(
    "numeric_encoding,truth", [("scalar", False), ("scalar", True), ("bits", True)]
)
def test_ensemble_guard_and_tuning_marker_preserve_current_resume(
    tmp_path, monkeypatch, encoding, numeric_encoding, truth
):
    task = SimpleNamespace(db_name="db", table_name="task", task_type="reg")
    calls = []

    class Evaluator:
        def evaluate_raw(self, _models, ctxs):
            calls.append(True)
            for ctx in ctxs:
                yield task, ctx, np.array([0.0, 1.0]), {"": np.array([0.1, 0.9])}, None

    monkeypatch.setattr(_eval, "build_evaluator", lambda *a, **k: Evaluator())
    args = {
        "grid": [(2, 1, True)],
        "ctx_sizes": [2, 4],
        "val_items": 2,
        "test_items": 2,
        "context_seed": 0,
        "val_ensemble_size": 1,
        "test_ensemble_size": 1,
        "tune_only": True,
        "tuning_out_path": tmp_path / "tuning.json",
        "resume_path": tmp_path / "resume.pt",
        "csv_out_dir": None,
        "targets": {},
        "use_wandb": False,
        "embedder": "unused",
        "shuffle_seed": 0,
        "db_cutoff": None,
        "exact_datetime": encoding == "calendar",
        "exact_numeric": truth,
    }
    model = SimpleNamespace(
        datetime_encoding=encoding, numeric_encoding=numeric_encoding
    )
    _eval.run_ensemble(model, "unused", [task], [], **args)
    state = torch.load(args["resume_path"], weights_only=False)
    assert (V2 in state["guard"]) == (encoding == "calendar")
    expected_suffix = (
        ([V2] if encoding == "calendar" else [])
        + (["numeric_fp32_truth_v1"] if truth else [])
        + (["numeric_bits_v1"] if numeric_encoding == "bits" else [])
    )
    assert state["guard"][11:] == expected_suffix
    rec = json.loads(args["tuning_out_path"].read_text())["db/task"]
    assert rec[KEY] == (V2 if encoding == "calendar" else "scalar")
    assert rec["numeric_encoding"] == numeric_encoding
    assert rec["exact_numeric"] == truth
    _eval.run_ensemble(model, "unused", [task], [], **args)
    assert len(calls) == 1  # Matching version skips already-completed tuning.
    if encoding == "calendar":
        for old in ("calendar_datetime_v1", None):
            guard = list(state["guard"])
            if old is None:
                guard.remove(V2)
            else:
                guard[guard.index(V2)] = old
            stale = dict(state, guard=guard)
            torch.save(stale, args["resume_path"])
            before = args["resume_path"].read_bytes()
            with pytest.raises(AssertionError, match="different evaluation"):
                _eval.run_ensemble(model, "unused", [task], [], **args)
            assert args["resume_path"].read_bytes() == before
        assert len(calls) == 1
    for marker in ("numeric_fp32_truth_v1", "numeric_bits_v1"):
        if marker in state["guard"]:
            guard = [value for value in state["guard"] if value != marker]
        else:
            guard = [*state["guard"], marker]
        torch.save(dict(state, guard=guard), args["resume_path"])
        with pytest.raises(AssertionError, match="different evaluation"):
            _eval.run_ensemble(model, "unused", [task], [], **args)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "encoding,marker,accepted",
    [
        ("calendar", None, False),
        ("calendar", "calendar_datetime_v1", False),
        ("calendar", V2, True),
        ("scalar", None, True),
    ],
)
def test_both_planners_check_completed_results(
    tmp_path, monkeypatch, encoding, marker, accepted
):
    ckpt = checkpoint(tmp_path, encoding)
    monkeypatch.setattr(tune.env, "out_root", lambda: str(tmp_path))
    monkeypatch.setattr(tune.env, "pre_dir", lambda: "unused")
    monkeypatch.setattr(
        ensemble,
        "load_configs",
        lambda *a: {"db/task": {"top_cfgs": [[2, 2, 1, True]]}},
    )
    grid = "tune-test--{db}--{table}"
    tuning = tune.grid_path(grid, "db", "task")
    final = tmp_path / "results/cfg0/db__task.json"
    tuning.parent.mkdir(parents=True)
    final.parent.mkdir(parents=True)
    stamp = {} if marker is None else {KEY: marker}
    tuning.write_text(json.dumps({"db/task": stamp}))
    final.write_text(json.dumps({"config": stamp}))
    args = {"ckpt": ckpt, "grid_stem": grid, "task_list": [("db", "task")]}
    for fn, kw, path in [
        (tune.jobs, args, tuning),
        (
            ensemble.jobs,
            args | {"configs_path": "unused", "out_subdir": "results"},
            final,
        ),
    ]:
        before = path.read_bytes()
        if accepted:
            assert fn(**kw) == []
        else:
            with pytest.raises(ValueError, match="datetime context version"):
                fn(**kw)
        assert path.read_bytes() == before


def test_fresh_plans_do_not_resolve_or_load_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(tune.env, "out_root", lambda: str(tmp_path))
    monkeypatch.setattr(tune.env, "pre_dir", lambda: "unused")
    monkeypatch.setattr(
        ensemble,
        "load_configs",
        lambda *a: {"db/task": {"top_cfgs": [[2, 2, 1, True]]}},
    )
    for module in (tune, ensemble):
        monkeypatch.setattr(module, "checkpoint_datetime_cache_version", no_model)
        monkeypatch.setattr(module, "_numeric_settings", no_model)
    args = {
        "ckpt": "unresolved",
        "grid_stem": "tune-test--{db}--{table}",
        "task_list": [("db", "task")],
    }
    assert len(tune.jobs(**args)) == 1
    assert len(ensemble.jobs(**args, configs_path="unused", out_subdir="results")) == 1


def completed_calls(tmp_path, monkeypatch, ckpt, exact_numeric=False):
    monkeypatch.setattr(tune.env, "out_root", lambda: str(tmp_path))
    monkeypatch.setattr(tune.env, "pre_dir", lambda: "unused")
    monkeypatch.setattr(
        ensemble,
        "load_configs",
        lambda *a: {"db/task": {"top_cfgs": [[2, 2, 1, True]]}},
    )
    grid = "combined--{db}--{table}"
    tuning = tune.grid_path(grid, "db", "task")
    final = tmp_path / "db__task.json"
    planned = tmp_path / "results/cfg0/db__task.json"
    for path in (tuning, final, planned):
        path.parent.mkdir(parents=True, exist_ok=True)
    args = {
        "ckpt": ckpt,
        "grid_stem": grid,
        "task_list": [("db", "task")],
        "exact_numeric": exact_numeric,
    }
    calls = [
        lambda: run.main(**worker_args(tmp_path, ckpt), exact_numeric=exact_numeric),
        lambda: tune.jobs(**args),
        lambda: ensemble.jobs(**args, configs_path="unused", out_subdir="results"),
    ]
    return (tuning, final, planned), calls


def write_completed(paths, stamp):
    tuning, final, planned = paths
    tuning.write_text(json.dumps({"db/task": stamp}))
    final.write_text(json.dumps({"config": stamp}))
    planned.write_bytes(final.read_bytes())


@pytest.mark.parametrize("encoding", ["scalar", "calendar"])
@pytest.mark.parametrize(
    "numeric_encoding,requested", [("scalar", False), ("scalar", True), ("bits", False)]
)
def test_combined_worker_and_planner_boundaries(
    tmp_path, monkeypatch, encoding, numeric_encoding, requested
):
    import rt.model

    monkeypatch.setattr(rt.model, "load_rt_model", no_model)
    ckpt = checkpoint(tmp_path, encoding, numeric_encoding=numeric_encoding)
    truth = requested or numeric_encoding == "bits"
    stamp = {
        KEY: V2 if encoding == "calendar" else "scalar",
        "numeric_encoding": numeric_encoding,
        "exact_numeric": truth,
    }
    paths, calls = completed_calls(tmp_path, monkeypatch, ckpt, requested)
    write_completed(paths, stamp)
    for call in calls:
        assert call() in (None, [])

    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()
    partial = partial_dir / "db__task.state.npz"
    partial_args = worker_args(partial_dir, ckpt) | {"exact_numeric": requested}
    for changed in (
        {KEY: "calendar_datetime_v1"},
        {"numeric_encoding": "scalar" if numeric_encoding == "bits" else "bits"},
        {"exact_numeric": not truth},
    ):
        stale = stamp | changed
        write_completed(paths, stale)
        before = [p.read_bytes() for p in paths]
        error = "datetime context version" if KEY in changed else "numeric encoding"
        for call in calls:
            with pytest.raises(ValueError, match=error):
                call()
        assert [p.read_bytes() for p in paths] == before
        # Complete partial, deliberately without accumulators: reject first.
        np.savez(partial, seeds=2, **stale)
        before = partial.read_bytes()
        with pytest.raises(ValueError, match=error):
            run.main(**partial_args)
        assert partial.read_bytes() == before
        assert not (partial_dir / "db__task.json").exists()

    write_completed(paths, {})
    for call in calls:
        if encoding == "scalar" and numeric_encoding == "scalar" and not truth:
            assert call() in (None, [])
        else:
            with pytest.raises(ValueError, match="different"):
                call()


@pytest.mark.parametrize(
    "encoding,width,tag,accepted",
    [
        ("scalar", 1, "none", True),
        ("scalar", 5, "none", False),
        ("scalar", 5, "config", False),
        ("bits", 5, "config", True),
        ("bits", 5, "file", True),
        ("bits", 1, "config", False),
    ],
)
def test_numeric_header_before_worker_and_planner_cache_hits(
    tmp_path, monkeypatch, encoding, width, tag, accepted
):
    import rt.model

    monkeypatch.setattr(rt.model, "load_rt_model", no_model)
    ckpt = checkpoint(
        tmp_path,
        "scalar",
        tag="none",
        numeric_encoding=encoding,
        numeric_width=width,
        numeric_tag=tag,
    )
    paths, calls = completed_calls(tmp_path, monkeypatch, ckpt)
    stamp = (
        {"numeric_encoding": encoding, "exact_numeric": True}
        if encoding == "bits"
        else {}
    )
    write_completed(paths, stamp)
    before = [p.read_bytes() for p in paths]
    if accepted:
        assert run._numeric_settings(ckpt, False) == (encoding, encoding == "bits")
        for call in calls:
            assert call() in (None, [])
    else:
        for call in [lambda: run._numeric_settings(ckpt, False), *calls]:
            with pytest.raises(ValueError, match="coordinate number adapter"):
                call()
    assert [p.read_bytes() for p in paths] == before
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("shape", [None, (2,), (2, 1, 1), (2, 0)])
def test_numeric_header_missing_or_malformed(tmp_path, shape):
    ckpt = checkpoint(tmp_path, "scalar", numeric_tag="none")
    tensors = {"enc_dict.datetime.weight": torch.zeros(2, 1)}
    if shape is not None:
        tensors["enc_dict.number.weight"] = torch.zeros(shape)
    save_file(tensors, tmp_path / "checkpoint/model.safetensors")
    with pytest.raises(ValueError, match="missing|1-coordinate number"):
        run._numeric_settings(ckpt, False)


def test_tuning_skip_requires_requested_task_and_retains_all_numeric_checks(
    tmp_path, monkeypatch
):
    ckpt = checkpoint(tmp_path, "scalar")
    paths, calls = completed_calls(tmp_path, monkeypatch, ckpt)
    paths[0].write_text(json.dumps({"other/task": {}}))
    with pytest.raises(KeyError, match="db/task"):
        calls[1]()
    paths[0].write_text(
        json.dumps({"db/task": {}, "other/task": {"exact_numeric": True}})
    )
    with pytest.raises(ValueError, match="numeric encoding"):
        calls[1]()
