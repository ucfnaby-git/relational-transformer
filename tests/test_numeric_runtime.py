import json
import os
import signal
import subprocess
import sys
from types import SimpleNamespace
from typing import ClassVar

import ml_dtypes
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import IterableDataset

from rt.data import datasets
from rt.eval import _eval
from rt.eval.evaluator import _numeric_labels
from rt.eval.metrics import metric_for
from rt.model.checkpoints import load_model, save_model
from rt.train import _train
from rt.train.swa import SwaState


def bf16(values):
    return np.asarray(values, dtype=ml_dtypes.bfloat16)


def native_fields(*, calendar=False, numeric=False):
    fields = {
        name: np.zeros(4, dtype=np.int32)
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
    fields.update(
        {
            name: np.zeros(4, dtype=bool)
            for name in ("is_targets", "is_task_nodes", "is_padding")
        }
    )
    fields.update(
        seq_len=2,
        f2p_nbr_idxs=np.full(20, -1, dtype=np.int32),
        number_values=bf16([1.0001, 0, 1.0002, 0]),
        datetime_values=bf16(np.zeros(4)),
        text_values=bf16(np.zeros(8)),
        col_name_values=bf16(np.zeros(8)),
    )
    if calendar:
        fields.update(
            datetime_features=bf16(np.zeros(56)),
            datetime_target_values=np.zeros(4, dtype=np.float32),
            datetime_seconds=np.zeros(4, dtype=np.int64),
            datetime_nanoseconds=np.zeros(4, dtype=np.int32),
            datetime_valid=np.zeros(4, dtype=bool),
        )
    if numeric:
        fields.update(
            number_features=bf16(np.arange(20) / 256),
            number_target_values=np.array([1.0001, 0, 1.0002, 0], dtype=np.float32),
        )
    return fields


@pytest.mark.parametrize("calendar", [False, True])
@pytest.mark.parametrize("numeric", [False, True])
def test_batch_modes_preserve_native_dtypes_and_legacy_values(calendar, numeric):
    fields = native_fields(calendar=calendar, numeric=numeric)
    batch = datasets.process_batch(fields.items(), d_text=2)
    legacy = datasets.process_batch(native_fields().items(), d_text=2)
    for key, value in legacy.items():
        assert torch.equal(batch[key], value)
    assert ("number_features" in batch) == numeric
    assert ("datetime_features" in batch) == calendar
    if numeric:
        assert batch["number_features"].dtype == torch.bfloat16
        assert batch["number_features"].shape == (2, 2, 5)
        targets = batch["number_target_values"]
        assert targets.dtype == torch.float32 and targets.shape == (2, 2, 1)
        assert torch.equal(
            targets.flatten(), torch.from_numpy(fields["number_target_values"])
        )
        assert targets[0, 0] != targets[1, 0]
        assert batch["number_values"][0, 0] == batch["number_values"][1, 0]


@pytest.mark.parametrize("dtype", [np.float16, np.uint16, np.float32])
def test_numeric_feature_dtype_is_not_inferred_from_width(dtype):
    fields = native_fields(numeric=True)
    fields["number_features"] = np.zeros(20, dtype=dtype)
    with pytest.raises(ValueError, match="native BF16 dtype"):
        datasets.process_batch(fields.items(), d_text=2)


@pytest.mark.parametrize("missing", ["number_features", "number_target_values"])
def test_partial_numeric_payload_rejected(missing):
    fields = native_fields(numeric=True)
    del fields[missing]
    with pytest.raises(ValueError, match="features and FP32 targets"):
        datasets.process_batch(fields.items(), d_text=2)


def test_numeric_target_dtype_and_alignment_rejected():
    fields = native_fields(numeric=True)
    fields["number_target_values"] = np.zeros(4, dtype=np.float64)
    with pytest.raises(ValueError, match="source FP32"):
        datasets.process_batch(fields.items(), d_text=2)
    fields["number_target_values"] = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="align"):
        datasets.process_batch(fields.items(), d_text=2)


@pytest.mark.parametrize("calendar", [False, True])
@pytest.mark.parametrize("numeric", [False, True])
def test_dataset_flags_are_independent_and_legacy_native_kwargs_unchanged(
    monkeypatch, tmp_path, calendar, numeric
):
    captured = {}

    def sampler(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(num_items=2)

    monkeypatch.setattr(datasets, "Sampler", sampler)
    ds = datasets.TrainDataset(
        tasks=[],
        pre_dir=str(tmp_path),
        train_ctx_size_list=[2],
        train_tokens_per_gpu=4,
        total_bs=2,
        global_rank=0,
        local_rank=0,
        world_size=1,
        local_ctx_size_list=[2],
        bfs_width_list=[1],
        num_walks=1,
        walk_length=1,
        prefer_latest_list=[True],
        mask_prob_max=0,
        embedder="unused",
        d_text=2,
        seed=0,
        items_per_task=2,
        mask_prob_max_shared=None,
        mmap_populate=False,
        timeout_per_item=1,
        vector_db_path=None,
        db_cutoff=None,
        exact_datetime=calendar,
        exact_numeric=numeric,
    )
    assert captured.get("exact_datetime", False) == calendar
    assert captured.get("exact_numeric", False) == numeric
    assert ("exact_numeric" in captured) == numeric
    assert ("exact_datetime" in captured) == calendar
    ds._process_batch(native_fields(calendar=calendar, numeric=numeric).items())
    if numeric:
        with pytest.raises(ValueError, match="requires native numeric"):
            ds._process_batch(native_fields(calendar=calendar).items())


def label_batch():
    # Masked NaN/inf must not poison real targets or dropped rows.
    batch = {
        "number_values": torch.tensor([[1, 0], [1, 0], [0, 0]], dtype=torch.bfloat16),
        "number_target_values": torch.tensor(
            [
                [[1.0001], [float("nan")]],
                [[1.0002], [float("inf")]],
                [[float("nan")], [float("nan")]],
            ],
            dtype=torch.float32,
        ),
        "is_targets": torch.tensor([[True, False], [True, False], [False, False]]),
        "is_padding": torch.tensor([[False, False], [False, False], [True, True]]),
        "sem_types": torch.zeros((3, 2), dtype=torch.int32),
        "node_idxs": torch.tensor([[1, 2], [3, 4], [0, 0]], dtype=torch.int32),
        "col_name_idxs": torch.zeros((3, 2), dtype=torch.int32),
        "is_task_nodes": torch.ones((3, 2), dtype=torch.bool),
    }
    return batch, torch.tensor([True, True, False])


def test_common_fp32_truth_and_legacy_fallback():
    batch, mask = label_batch()
    expected = batch["number_target_values"][:2, 0, 0]
    labels = _numeric_labels(batch, mask)
    assert labels.dtype == torch.float32
    assert torch.equal(labels[:2], expected) and labels[2] == 0
    assert labels[:2].unique().numel() == 2
    del batch["number_target_values"]
    legacy = _numeric_labels(batch, mask)
    assert legacy.dtype == torch.bfloat16
    assert torch.equal(legacy, torch.tensor([1, 1, 0], dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="requires number_target_values"):
        _numeric_labels(batch, mask, require_exact=True)


def test_gather_preserves_signed_zero_and_subnormal_bits():
    bits = torch.tensor([-2147483648, 1, 0x007FFFFF], dtype=torch.int32)
    batch, _ = label_batch()
    batch["number_target_values"] = torch.zeros((3, 2, 1))
    batch["number_target_values"][:, 0, 0] = bits.view(torch.float32)
    batch["is_targets"][:, 0] = True
    batch["is_padding"][:] = False
    labels = _numeric_labels(batch, torch.ones(3, dtype=torch.bool))
    assert torch.equal(labels.view(torch.int32), bits)


@pytest.mark.parametrize("flush", [False, True])
def test_label_bit_transport_in_isolated_flush_modes(flush):
    code = f"""
import torch
from rt.eval.evaluator import _numeric_labels
torch.set_num_threads(1)
assert torch.set_flush_denormal({flush!r})
bits = torch.tensor([-2147483648, 1, 0x007fffff], dtype=torch.int32)
batch = {{
    'number_target_values': bits.view(torch.float32).reshape(3, 1, 1),
    'is_targets': torch.ones((3, 1), dtype=torch.bool),
    'is_padding': torch.zeros((3, 1), dtype=torch.bool),
    'sem_types': torch.zeros((3, 1), dtype=torch.int32),
}}
labels = _numeric_labels(batch, torch.ones(3, dtype=torch.bool))
assert torch.equal(labels.view(torch.int32), bits)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        timeout=30,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
    )


@pytest.mark.parametrize("semantic_type", [1, 2, 3])
def test_exact_numeric_truth_rejects_nonnumber_primary(semantic_type):
    batch, mask = label_batch()
    batch["sem_types"][0, 0] = semantic_type
    with pytest.raises(ValueError, match="Number targets"):
        _numeric_labels(batch, mask)
    del batch["number_target_values"]
    assert _numeric_labels(batch, mask).dtype == torch.bfloat16


@pytest.mark.parametrize("problem", ["dtype", "shape", "none", "multiple", "infinite"])
def test_exact_numeric_truth_rejects_bad_targets(problem):
    batch, mask = label_batch()
    if problem == "dtype":
        batch["number_target_values"] = batch["number_target_values"].bfloat16()
    elif problem == "shape":
        batch["number_target_values"] = batch["number_target_values"].squeeze(-1)
    elif problem == "none":
        batch["is_targets"][0] = False
    elif problem == "multiple":
        batch["is_targets"][0] = True
    else:
        batch["number_target_values"][0, 0, 0] = float("inf")
    with pytest.raises(ValueError):
        _numeric_labels(batch, mask)


@pytest.mark.parametrize("ddp", [False, True])
@pytest.mark.parametrize("exact_numeric", [False, True])
def test_actual_evaluator_common_truth_and_gather_dtype(
    monkeypatch, ddp, exact_numeric
):
    from rt.data.tasks import Task
    from rt.eval import evaluator as module

    batch, mask = label_batch()
    captured = []

    class FixtureDataset:
        world_size = 2 if ddp else 1
        num_items = 3 * world_size

        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.sampler = SimpleNamespace(batch_py=self.emit)

        def emit(self, *_args):
            return {k: v.clone() for k, v in batch.items()} | {
                "batch_mask": mask.clone()
            }

        def _process_batch(self, value):
            return value

    class FixedPrediction:
        def eval(self):
            pass

        def predict(self, batch, contexts, device, task):
            return {ctx: torch.ones(3) for ctx in contexts}

    gathered_dtypes = []

    def gather(out, value):
        gathered_dtypes.append(value.dtype)
        out.copy_(value.repeat(2))

    monkeypatch.setattr(module, "RustlerDataset", FixtureDataset)
    monkeypatch.setattr(module.dist, "all_gather_single", gather)
    task = Task("fixture", "events", "target", "reg", "val")
    evaluator = _eval.build_evaluator(
        [task],
        "unused",
        embedder="unused",
        d_text=2,
        device="cpu",
        ctx_size_list=[2],
        local_ctx_size=2,
        bfs_width=1,
        num_walks=1,
        walk_length=1,
        tokens_per_gpu=6,
        items_per_task=6 if ddp else 3,
        num_workers=0,
        context_seed=0,
        prefer_latest=True,
        shuffle_seed=0,
        mmap_populate=False,
        prefetch_factor=1,
        vector_db_path=None,
        db_cutoff=None,
        world_size=2 if ddp else 1,
        ddp=ddp,
        exact_numeric=exact_numeric,
    )
    records = list(
        evaluator.evaluate_raw(
            [(FixedPrediction(), "scalar"), (FixedPrediction(), "bits")], [2]
        )
    )
    assert captured[0]["exact_numeric"] == exact_numeric
    assert len(records) == 1
    _, _, labels, preds, _ = records[0]
    expected = batch["number_target_values"][:2, 0, 0].numpy()
    assert labels.dtype == np.float32
    assert np.array_equal(labels, np.tile(expected, 2 if ddp else 1))
    assert metric_for("reg", labels, preds["scalar"]) == metric_for(
        "reg", labels, preds["bits"]
    )
    if ddp:
        assert gathered_dtypes[0] == torch.float32


class NumericNet(nn.Module):
    def __init__(
        self, *, datetime_encoding="scalar", numeric_encoding="scalar", **kwargs
    ):
        super().__init__()
        self.datetime_encoding = datetime_encoding
        self.numeric_encoding = numeric_encoding
        self.enc_dict = nn.ModuleDict(
            {
                "number": nn.Linear(5 if numeric_encoding == "bits" else 1, 2),
                "datetime": nn.Linear(14 if datetime_encoding == "calendar" else 1, 2),
            }
        )
        self.body = nn.Linear(2, 2, bias=False)
        self.dec_dict = nn.ModuleDict(
            {"number": nn.Linear(2, 1), "datetime": nn.Linear(2, 1)}
        )

    def set_compute_dtype(self, dtype=torch.bfloat16):
        self.enc_dict.to(dtype=dtype)
        self.body.to(dtype=dtype)
        for name, exact in (
            ("number", self.numeric_encoding == "bits"),
            ("datetime", self.datetime_encoding == "calendar"),
        ):
            self.dec_dict[name].to(dtype=torch.float32 if exact else dtype)
        return self

    def forward(self, batch, return_embeddings=False):
        values = (
            batch["number_features"]
            if self.numeric_encoding == "bits"
            else batch["number_values"]
        )
        hidden = self.body(self.enc_dict["number"](values))
        head = self.dec_dict["number"]
        prediction = head(hidden.to(head.weight.dtype)).float()
        return ((prediction - batch["y"]).square().mean(),)


class NumericReplay(IterableDataset):
    num_items = 2
    modes: ClassVar[list[tuple[bool, bool]]] = []

    def __init__(self, *, exact_datetime, exact_numeric, **kwargs):
        self.modes.append((exact_datetime, exact_numeric))

    def __iter__(self):
        while True:
            yield {
                "number_values": torch.tensor([[0.5], [-0.5]], dtype=torch.bfloat16),
                "number_features": torch.tensor(
                    [[0.5, 0.25, 0, 0, 0], [-0.5, 0.125, 0, 0, 0]], dtype=torch.bfloat16
                ),
                "y": torch.tensor([[0.12345], [-0.125]]),
            }


@pytest.fixture
def numeric_trainer(monkeypatch, tmp_path):
    from examples.pretrain.phases import phase_one

    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGUSR1)}
    NumericReplay.modes = []
    monkeypatch.setattr(_train, "setup_dist", lambda _: ("cpu", 0, 0, 1, False))
    monkeypatch.setattr(_train, "RelationalTransformer", NumericNet)
    monkeypatch.setattr(_train, "TrainDataset", NumericReplay)
    monkeypatch.setattr(
        _train,
        "get_tasks",
        lambda _p, _t, splits: ["fixture"] if splits == ("train",) else [],
    )
    args = phase_one(
        pre_dir="unused",
        eval_pre_dir="unused",
        out_root=str(tmp_path),
        project="numeric",
        run_id="run",
        stage_dir=None,
        tokens_per_gpu=2,
        num_workers=0,
        eval_tokens_per_gpu=2,
        eval_num_workers=0,
        total_steps=4,
        early_stop_after_steps=None,
    ).args
    args.update(
        datetime_encoding=None,
        numeric_encoding=None,
        compile=False,
        num_blocks=1,
        d_model=2,
        d_ff=4,
        d_text=2,
        num_heads=1,
        ctx_size_list=[1],
        eval_ctx_size_list=[1],
        local_ctx_size_list=[1],
        total_bs=4,
        optimizer="adamw",
        lr=0.01,
        lr_warmup_steps=0,
        eval_freq=None,
        swa_momentum=0.9,
        resume_save_mins=60,
    )
    try:
        yield args, tmp_path / "no-wandb-entity" / "numeric"
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


@pytest.mark.parametrize("date", ["scalar", "calendar"])
@pytest.mark.parametrize("number", ["scalar", "bits"])
def test_authoritative_warm_start_masters_both_modes(numeric_trainer, date, number):
    args, root = numeric_trainer
    net = NumericNet(datetime_encoding=date, numeric_encoding=number)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.fill_(1.0001)
    source = root / "source"
    source.mkdir(parents=True)
    (source / "config.json").write_text(
        json.dumps({"model": {"datetime_encoding": date, "numeric_encoding": number}})
    )
    save_model(net.state_dict(), source / "model.safetensors")
    _train.main(**(args | {"load_ckpt_path": str(source), "total_steps": 0}))
    resume = torch.load(root / "run/resume.pt", weights_only=True)
    assert resume["numeric_encoding"] == number
    assert resume["exact_numeric"] == (number == "bits")
    assert NumericReplay.modes == [(date == "calendar", number == "bits")]
    for name, original in net.named_parameters():
        assert torch.equal(resume["master"][name], original)
        assert torch.equal(resume["swa"]["params"][name], original)
    live = load_model(root / "run/steps=0.safetensors")
    assert live["dec_dict.number.weight"].dtype == (
        torch.float32 if number == "bits" else torch.bfloat16
    )
    assert live["dec_dict.datetime.weight"].dtype == (
        torch.float32 if date == "calendar" else torch.bfloat16
    )
    assert live["body.weight"].dtype == torch.bfloat16


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal_state(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("date", ["scalar", "calendar"])
@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_numeric_actual_trainer_resume_master_swa_exact(
    numeric_trainer, monkeypatch, date, optimizer
):
    args, root = numeric_trainer
    args.update(datetime_encoding=date, numeric_encoding="bits", optimizer=optimizer)
    _train.main(**(args | {"run_id": "continuous"}))

    class PreemptSwa(SwaState):
        def update(self, named):
            super().update(named)
            if self.n == 2:
                signal.raise_signal(signal.SIGUSR1)

    monkeypatch.setattr(_train, "SwaState", PreemptSwa)
    _train.main(**args)
    monkeypatch.setattr(_train, "SwaState", SwaState)
    _train.main(**(args | {"numeric_encoding": None, "datetime_encoding": None}))
    continuous = torch.load(root / "continuous/resume.pt", weights_only=True)
    resumed = torch.load(root / "run/resume.pt", weights_only=True)
    equal_state(continuous, resumed)
    assert resumed["step"] == resumed["swa"]["n"] == 4
    assert all(value.dtype == torch.float32 for value in resumed["master"].values())
    assert not torch.equal(
        resumed["master"]["dec_dict.number.weight"],
        resumed["swa"]["params"]["dec_dict.number.weight"],
    )
    saved = load_model(root / "run/swa_steps=4.safetensors")
    assert saved["dec_dict.number.weight"].dtype == torch.float32
    assert torch.equal(
        saved["dec_dict.number.weight"],
        resumed["swa"]["params"]["dec_dict.number.weight"],
    )


def test_scalar_exact_payload_and_resume_truth_guard(numeric_trainer):
    args, root = numeric_trainer
    _train.main(**(args | {"total_steps": 0, "exact_numeric": True}))
    resume = torch.load(root / "run/resume.pt", weights_only=True)
    assert resume["numeric_encoding"] == "scalar" and resume["exact_numeric"] is True
    assert NumericReplay.modes == [(False, True)]
    with pytest.raises(ValueError, match="resume evaluation truth"):
        _train.main(**args)


@pytest.mark.parametrize(
    "requested,saved,expected",
    [
        (None, None, "scalar"),
        (None, "scalar", "scalar"),
        (None, "bits", "bits"),
        ("bits", None, "bits"),
    ],
)
def test_numeric_encoding_inference(requested, saved, expected):
    assert _train._resolve_numeric_encoding(requested, saved) == expected


@pytest.mark.parametrize(
    "requested,saved",
    [("bits", "scalar"), ("scalar", "bits"), ("other", None), (None, "other")],
)
def test_numeric_encoding_conflicts_rejected(requested, saved):
    with pytest.raises(ValueError):
        _train._resolve_numeric_encoding(requested, saved)


@pytest.mark.parametrize("delta", [False, True])
def test_config_only_bits_rejects_bf16_head_before_widening(numeric_trainer, delta):
    args, root = numeric_trainer
    source = root / "source"
    source.mkdir(parents=True)
    (source / "config.json").write_text(
        json.dumps({"model": {"numeric_encoding": "bits"}})
    )
    state = NumericNet(numeric_encoding="bits").state_dict()
    save_model(state, source / "model.safetensors")
    args.update(load_ckpt_path=str(source), total_steps=0, delta_finetune=delta)
    if delta:
        _train.main(**args)
    state["dec_dict.number.weight"] = state["dec_dict.number.weight"].bfloat16()
    save_model(state, source / "model.safetensors")
    with pytest.raises(ValueError, match="FP32.*number"):
        _train.main(**args)


def test_recipe_jobs_forward_numeric_mode_and_truth():
    from examples.pretrain.phases import jobs

    args = {
        "plurel_pre_dir": "pl",
        "the_join_pre_dir": "join",
        "eval_pre_dir": "eval",
        "out_root": "out",
        "project": "test",
        "phase_one_run_id": "one",
        "phase_two_run_id": "two",
        "phase_two_load_ckpt_path": "checkpoint",
        "stage_dir": None,
        "tokens_per_gpu": 2,
        "num_workers": 0,
        "eval_tokens_per_gpu": 2,
        "eval_num_workers": 0,
        "total_steps": 4,
        "early_stop_after_steps": None,
    }
    for job in jobs(**args):
        assert job.args["numeric_encoding"] == "scalar"
        assert job.args["exact_numeric"] is False
    for job in jobs(
        **args,
        datetime_encoding="calendar",
        numeric_encoding="bits",
        exact_numeric=True,
    ):
        assert job.args["numeric_encoding"] == "bits"
        assert job.args["datetime_encoding"] == "calendar"
        assert job.args["exact_numeric"] is True


@pytest.mark.parametrize("problem", ["dtype", "shape"])
def test_resume_rejects_narrowed_or_misshaped_swa(numeric_trainer, problem):
    args, root = numeric_trainer
    args.update(numeric_encoding="bits", total_steps=0)
    _train.main(**args)
    path = root / "run/resume.pt"
    state = torch.load(path, weights_only=True)
    head = state["swa"]["params"]["dec_dict.number.weight"]
    state["swa"]["params"]["dec_dict.number.weight"] = (
        head.bfloat16() if problem == "dtype" else head.T.contiguous()
    )
    torch.save(state, path)
    with pytest.raises(ValueError, match="FP32 SWA state"):
        _train.main(**args)


def test_icl_numeric_state_guard_preserves_legacy_and_rejects_truth_changes():
    from examples.icl.run import _check_numeric_state

    _check_numeric_state({}, "scalar", False)
    _check_numeric_state(
        {"numeric_encoding": np.array("bits"), "exact_numeric": np.array(True)},
        "bits",
        True,
    )
    for state, encoding, exact in [
        ({}, "scalar", True),
        ({}, "bits", True),
        ({"numeric_encoding": "bits", "exact_numeric": True}, "scalar", True),
    ]:
        with pytest.raises(
            ValueError, match="different numeric encoding/evaluation truth"
        ):
            _check_numeric_state(state, encoding, exact)


def test_icl_job_apis_forward_explicit_common_truth(monkeypatch, tmp_path):
    from examples.icl import ensemble, tune

    monkeypatch.setattr(tune.env, "out_root", lambda: str(tmp_path))
    monkeypatch.setattr(tune.env, "pre_dir", lambda: "unused")
    monkeypatch.setattr(
        ensemble,
        "load_configs",
        lambda *_: {"db/task": {"top_cfgs": [[2, 2, 1, True]]}},
    )
    tuning = tune.jobs(
        ckpt="checkpoint",
        grid_stem="tune-test--{db}--{table}",
        task_list=[("db", "task")],
        exact_numeric=True,
    )
    testing = ensemble.jobs(
        ckpt="checkpoint",
        configs_path="unused",
        grid_stem="unused",
        task_list=[("db", "task")],
        out_subdir="results",
        exact_numeric=True,
    )
    assert tuning[0].args["exact_numeric"] is True
    assert testing[0].args["exact_numeric"] is True


def test_icl_planners_validate_completed_result_truth(monkeypatch, tmp_path):
    from examples.icl import ensemble, tune

    monkeypatch.setattr(tune.env, "out_root", lambda: str(tmp_path))
    monkeypatch.setattr(tune.env, "pre_dir", lambda: "unused")
    monkeypatch.setattr(
        ensemble,
        "load_configs",
        lambda *_: {"db/task": {"top_cfgs": [[2, 2, 1, True]]}},
    )
    for module in (tune, ensemble):
        monkeypatch.setattr(module, "_numeric_settings", lambda *_: ("bits", True))
    grid = "tune-test--{db}--{table}"
    tuning_path = tune.grid_path(grid, "db", "task")
    tuning_path.parent.mkdir(parents=True)
    testing_path = tmp_path / "results/cfg0/db__task.json"
    testing_path.parent.mkdir(parents=True)
    tuning_path.write_text(json.dumps({"db/task": {}}))
    testing_path.write_text(json.dumps({"config": {}}))
    tuning_args = {"ckpt": "unused", "grid_stem": grid, "task_list": [("db", "task")]}
    testing_args = tuning_args | {"configs_path": "unused", "out_subdir": "results"}
    for function, args in ((tune.jobs, tuning_args), (ensemble.jobs, testing_args)):
        with pytest.raises(
            ValueError, match="different numeric encoding/evaluation truth"
        ):
            function(**args)
    settings = {"numeric_encoding": "bits", "exact_numeric": True}
    tuning_path.write_text(json.dumps({"db/task": settings}))
    testing_path.write_text(json.dumps({"config": settings}))
    assert tune.jobs(**tuning_args) == []
    assert ensemble.jobs(**testing_args) == []


@pytest.mark.parametrize("encoding", ["scalar", "bits"])
def test_ensemble_resume_rejects_legacy_truth_cache(tmp_path, encoding):
    path = tmp_path / "resume.pt"
    legacy_guard = [[(2, 1, True)], [2], 2, 2, 0, 1, 1, True, [], 0, None]
    torch.save({"guard": legacy_guard}, path)
    with pytest.raises(AssertionError, match="different evaluation"):
        _eval.run_ensemble(
            SimpleNamespace(numeric_encoding=encoding),
            "unused",
            [],
            [],
            grid=[(2, 1, True)],
            ctx_sizes=[2],
            val_items=2,
            test_items=2,
            context_seed=0,
            val_ensemble_size=1,
            test_ensemble_size=1,
            tune_only=True,
            tuning_out_path=tmp_path / "tuning.json",
            resume_path=path,
            csv_out_dir=None,
            targets={},
            use_wandb=False,
            embedder="unused",
            shuffle_seed=0,
            db_cutoff=None,
            exact_numeric=True,
        )
