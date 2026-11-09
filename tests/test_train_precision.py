import contextlib
import json
import signal
from datetime import timedelta
from typing import ClassVar

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import IterableDataset

from rt.model import RelationalTransformer
from rt.model.checkpoints import load_model, save_model
from rt.train import _train
from rt.train.swa import SwaState


class TinyNet(nn.Module):
    """Exercise the trainer's dtype/lifecycle contract without GPU attention."""

    def __init__(self, *, datetime_encoding="scalar", **kwargs):
        super().__init__()
        self.datetime_encoding = datetime_encoding
        self.body = nn.Linear(2, 2, bias=False)
        self.dec_dict = nn.ModuleDict({"datetime": nn.Linear(2, 1)})

    def set_compute_dtype(self, dtype=torch.bfloat16):
        self.body.to(dtype=dtype)
        self.dec_dict["datetime"].to(
            dtype=torch.float32 if self.datetime_encoding == "calendar" else dtype
        )
        return self

    def forward(self, batch, return_embeddings=False):
        x = self.body(batch["x"])
        head = self.dec_dict["datetime"]
        pred = head(x.to(head.weight.dtype)).float()
        return ((pred - batch["y"]).square().mean(),)


class ReplayDataset(IterableDataset):
    num_items = 2
    exact_modes: ClassVar[list[bool]] = []

    def __init__(self, *, exact_datetime, **kwargs):
        self.exact_modes.append(exact_datetime)

    def __iter__(self):
        while True:
            yield {
                "x": torch.tensor([[0.5, 0.25], [0.25, -0.5]], dtype=torch.bfloat16),
                "y": torch.tensor([[0.12345], [-0.125]]),
            }


@pytest.fixture
def trainer(monkeypatch, tmp_path):
    from examples.pretrain.phases import phase_one

    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGUSR1)}
    ReplayDataset.exact_modes = []
    monkeypatch.setattr(_train, "setup_dist", lambda _: ("cpu", 0, 0, 1, False))
    monkeypatch.setattr(_train, "RelationalTransformer", TinyNet)
    monkeypatch.setattr(_train, "TrainDataset", ReplayDataset)
    monkeypatch.setattr(
        _train,
        "get_tasks",
        lambda _p, _t, splits: ["fixture"] if splits == ("train",) else [],
    )
    args = phase_one(
        pre_dir="unused",
        eval_pre_dir="unused",
        out_root=str(tmp_path),
        project="precision",
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
        yield args, tmp_path / "no-wandb-entity" / "precision"
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


@pytest.mark.parametrize(
    ("requested", "saved", "expected"),
    [
        (None, None, "scalar"),
        (None, "scalar", "scalar"),
        (None, "calendar", "calendar"),
        ("calendar", None, "calendar"),
        ("scalar", "scalar", "scalar"),
        ("calendar", "calendar", "calendar"),
    ],
)
def test_encoding_resolution(requested, saved, expected):
    assert _train._resolve_datetime_encoding(requested, saved) == expected


@pytest.mark.parametrize(
    ("requested", "saved"), [("calendar", "scalar"), ("scalar", "calendar")]
)
def test_schema_changes_require_explicit_migration(requested, saved):
    with pytest.raises(ValueError, match="explicitly migrate"):
        _train._resolve_datetime_encoding(requested, saved)


def test_unknown_encoding_rejected():
    with pytest.raises(ValueError, match="unknown datetime_encoding"):
        _train._resolve_datetime_encoding("other")


@pytest.mark.parametrize("encoding", ["scalar", "calendar"])
def test_fp32_warm_start_is_authoritative(trainer, encoding):
    args, root = trainer
    initial = TinyNet(datetime_encoding=encoding)
    with torch.no_grad():
        for p in initial.parameters():
            p.fill_(1.0001)
    source = root / "source"
    source.mkdir(parents=True)
    model_config = {} if encoding == "scalar" else {"datetime_encoding": encoding}
    (source / "config.json").write_text(json.dumps({"model": model_config}))
    save_model(initial.state_dict(), source / "model.safetensors")
    _train.main(**(args | {"load_ckpt_path": str(source), "total_steps": 0}))
    ck = torch.load(root / "run" / "resume.pt", weights_only=True)
    assert ck["datetime_encoding"] == encoding
    assert ReplayDataset.exact_modes == [encoding == "calendar"]
    for name, original in initial.named_parameters():
        assert torch.equal(ck["master"][name], original)
        assert torch.equal(ck["swa"]["params"][name], original)
        assert not torch.equal(original, original.bfloat16().float())
    live = load_model(root / "run" / "steps=0.safetensors")
    expected_dtype = torch.float32 if encoding == "calendar" else torch.bfloat16
    assert live["dec_dict.datetime.weight"].dtype == expected_dtype
    assert live["body.weight"].dtype == torch.bfloat16
    config = json.loads((root / "run" / "config.json").read_text())
    assert config["model"]["datetime_encoding"] == encoding
    with pytest.raises(ValueError, match="does not match checkpoint"):
        _train.main(
            **(
                args
                | {
                    "run_id": "mismatch",
                    "load_ckpt_path": str(source),
                    "datetime_encoding": "scalar"
                    if encoding == "calendar"
                    else "calendar",
                }
            )
        )


@pytest.mark.parametrize("resume_delta", [False, True])
def test_config_only_calendar_rejects_narrowed_warm_start_head(trainer, resume_delta):
    args, root = trainer
    source = root / "source"
    source.mkdir(parents=True)
    (source / "config.json").write_text(
        json.dumps({"model": {"datetime_encoding": "calendar"}})
    )
    initial = TinyNet(datetime_encoding="calendar")
    state = initial.state_dict()
    save_model(state, source / "model.safetensors")  # Deliberately no file tag.
    run_args = args | {
        "load_ckpt_path": str(source),
        "total_steps": 0,
        "delta_finetune": resume_delta,
    }
    if resume_delta:
        # Resume skips the warm-start load, so exercise delta-base validation too.
        _train.main(**run_args)
    narrowed = {
        name: value.bfloat16() if name.startswith("dec_dict.datetime.") else value
        for name, value in state.items()
    }
    save_model(narrowed, source / "model.safetensors")
    with pytest.raises(ValueError, match="requires stored FP32 dec_dict.datetime"):
        _train.main(**run_args)


def assert_same_state(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_same_state(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_same_state(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("encoding", ["scalar", "calendar"])
@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_actual_trainer_resume_preserves_master_optimizer_and_swa(
    trainer, monkeypatch, encoding, optimizer
):
    args, root = trainer
    args["datetime_encoding"] = encoding
    args["optimizer"] = optimizer
    _train.main(**(args | {"run_id": "continuous"}))

    class PreemptSwa(SwaState):
        def update(self, named_tensors):
            super().update(named_tensors)
            if self.n == 2:
                signal.raise_signal(signal.SIGUSR1)

    monkeypatch.setattr(_train, "SwaState", PreemptSwa)
    _train.main(**args)
    interrupted = torch.load(root / "run" / "resume.pt", weights_only=True)
    assert interrupted["step"] == interrupted["swa"]["n"] == 2
    monkeypatch.setattr(_train, "SwaState", SwaState)
    _train.main(**(args | {"datetime_encoding": None}))
    continued = torch.load(root / "continuous" / "resume.pt", weights_only=True)
    resumed = torch.load(root / "run" / "resume.pt", weights_only=True)
    assert_same_state(continued, resumed)
    assert resumed["step"] == resumed["swa"]["n"] == 4
    assert all(t.dtype == torch.float32 for t in resumed["master"].values())
    for opt in resumed["optimizers"]:
        for state in opt["state"].values():
            if "momentum_buffer" in state:
                assert state["momentum_buffer"].dtype == torch.float32
            else:
                assert (
                    state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
                )
    swa_live = load_model(root / "run" / "swa_steps=4.safetensors")
    for name, tensor in swa_live.items():
        assert torch.equal(tensor, resumed["swa"]["params"][name].to(tensor.dtype))
    with pytest.raises(ValueError, match="does not match checkpoint"):
        _train.main(
            **(
                args
                | {
                    "datetime_encoding": "scalar"
                    if encoding == "calendar"
                    else "calendar"
                }
            )
        )


def test_untagged_resume_remains_scalar(trainer):
    args, root = trainer
    _train.main(**(args | {"total_steps": 0}))
    path = root / "run" / "resume.pt"
    ck = torch.load(path, weights_only=True)
    del ck["datetime_encoding"]
    torch.save(ck, path)
    _train.main(**(args | {"total_steps": 1}))
    assert torch.load(path, weights_only=True)["datetime_encoding"] == "scalar"
    assert ReplayDataset.exact_modes == [False, False]


@pytest.mark.parametrize("encoding", ["scalar", "calendar"])
def test_real_model_and_swa_compute_policy_preserve_fp32_head(encoding):
    kwargs = {
        "num_blocks": 1,
        "d_model": 8,
        "d_text": 4,
        "num_heads": 2,
        "d_ff": 16,
        "compile": False,
        "materialize_attn_masks": True,
        "datetime_encoding": encoding,
    }
    net = RelationalTransformer(**kwargs)
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(1.0001)
    masters = _train._init_master_params(net)
    net.set_compute_dtype(torch.bfloat16)
    swa = SwaState(masters.items(), momentum=0.9)
    swa_net = RelationalTransformer(**kwargs)
    swa_net.set_compute_dtype(torch.bfloat16)
    swa.sync_to(swa_net.named_parameters())
    for model in (net, swa_net):
        for name, p in model.named_parameters():
            expected = (
                torch.float32
                if encoding == "calendar" and name.startswith("dec_dict.datetime.")
                else torch.bfloat16
            )
            assert p.dtype == expected
            assert masters[name].dtype == torch.float32
            assert torch.equal(p, masters[name].to(expected))
            assert torch.equal(masters[name], torch.full_like(masters[name], 1.0001))


def test_master_state_rejects_wrong_names_shapes_and_dtype():
    net = TinyNet()
    state = {n: p.detach().clone() for n, p in net.named_parameters()}
    with pytest.raises(ValueError, match="names"):
        _train._init_master_params(net, {})
    for bad in (torch.zeros(1), state["body.weight"].bfloat16()):
        with pytest.raises(ValueError, match="expected FP32 master"):
            _train._init_master_params(net, state | {"body.weight": bad})


def _ddp_master_worker(rank, rendezvous, encoding):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(123 + rank)
        net = TinyNet(datetime_encoding=encoding)
        masters = _train._init_master_params(net, ddp=True)
        net.set_compute_dtype()
        with torch.no_grad():
            for n, p in net.named_parameters():
                p.copy_(masters[n])
        swa = SwaState(masters.items(), momentum=0.9)
        ddp = torch.nn.parallel.DistributedDataParallel(
            net, gradient_as_bucket_view=True
        )
        opt = torch.optim.SGD(masters.values(), lr=0.01)

        def assert_replicated(tensors):
            for t in tensors:
                values = [torch.empty_like(t) for _ in range(2)]
                dist.all_gather(values, t.detach())
                assert torch.equal(values[0], values[1])

        assert_replicated(masters.values())
        assert_replicated(net.parameters())
        assert_replicated(swa.params.values())
        batch = next(iter(ReplayDataset(exact_datetime=encoding == "calendar")))
        for _ in range(2):
            for micro in range(2):
                with ddp.no_sync() if micro == 0 else contextlib.nullcontext():
                    (ddp(batch)[0] / 2).backward()
            for n, p in net.named_parameters():
                masters[n].grad = p.grad.float()
            opt.step()
            with torch.no_grad():
                for n, p in net.named_parameters():
                    p.copy_(masters[n])
            swa.update(masters.items())
            assert_replicated(masters.values())
            assert_replicated(net.parameters())
            assert_replicated(swa.params.values())
            opt.zero_grad(set_to_none=True)
            net.zero_grad(set_to_none=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("encoding", ["scalar", "calendar"])
def test_two_rank_master_sync_before_live_and_swa(tmp_path, encoding):
    # Earlier tests may initialize threaded autograd; fork would inherit its
    # unsafe engine state. Spawn gives each rank a fresh interpreter instead.
    mp.start_processes(
        _ddp_master_worker,
        args=((tmp_path / "gloo").as_uri(), encoding),
        nprocs=2,
        start_method="spawn",
        join=True,
    )
