"""CPU contracts for bounded5 context and approximate FP32 numeric supervision.

The feature fixture mirrors the native wire format; native codec/transport tests
remain separate. Controlled projection tests do not prove learned injectivity.
"""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from test_datetime_model import batch as datetime_batch

from rt.model import checkpoints, net

DIMS = {"num_blocks": 0, "d_model": 4, "d_text": 2, "num_heads": 1, "d_ff": 8}


@pytest.fixture(autouse=True)
def cpu_attention(monkeypatch):
    monkeypatch.setattr(net, "_make_block_mask", lambda mask, **kwargs: mask)
    monkeypatch.setattr(
        net,
        "flex_attention",
        lambda q, k, v, *, block_mask, scale: F.scaled_dot_product_attention(
            q, k, v, attn_mask=block_mask[:, None], scale=scale
        ),
    )


def model(numeric="bits", datetime="scalar", dtype=torch.bfloat16, **kwargs):
    return net.RelationalTransformer(
        **{**DIMS, **kwargs},
        compile=False,
        materialize_attn_masks=True,
        numeric_encoding=numeric,
        datetime_encoding=datetime,
    ).set_compute_dtype(dtype)


def features(values):
    values = values.float().contiguous()
    bits = values.view(torch.int32).long() & 0xFFFFFFFF
    exponent = (bits >> 23) & 255
    safe = torch.where(exponent != 0, values, 0.0).double()
    cue = (safe / (1 + safe.abs())).bfloat16()
    digits = [((bits >> shift) & 255).float().div(256) for shift in (24, 16, 8, 0)]
    return torch.stack([cue, *digits], -1).bfloat16()


def batch():
    b = datetime_batch()
    labels = torch.tensor([[0.0, -0.2501234, 1.2501234, 0.0]])
    b["number_values"] = labels[..., None].bfloat16()
    b["number_target_values"] = labels[..., None]
    b["number_features"] = torch.where(
        (b["sem_types"] == 0)[..., None], features(labels), 0.0
    )
    return b


def assert_outputs_equal(a, b):
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[3], b[3])
    for name in net.SEM_TYPE_NAMES:
        assert torch.equal(a[1][name], b[1][name])
        assert torch.equal(a[2][name], b[2][name])


@pytest.mark.parametrize("datetime", ["scalar", "calendar"])
@pytest.mark.parametrize("loss_fn", ["huber", "l1", "bce"])
def test_scalar_numeric_ignores_extra_exact_payload(datetime, loss_fn):
    m = model(numeric="scalar", datetime=datetime, loss_fn=loss_fn)
    b = batch()
    b.pop("number_features")
    b.pop("number_target_values")
    expected = m(b, return_embeddings=False)
    # Even invalid extra fields must not silently change scalar model semantics.
    b["number_features"] = torch.full((1, 1, 99), float("nan"))
    b["number_target_values"] = torch.full((1,), float("nan"))
    assert_outputs_equal(expected, m(b, return_embeddings=False))


@pytest.mark.parametrize("numeric", ["scalar", "bits"])
@pytest.mark.parametrize("datetime", ["scalar", "calendar"])
def test_independent_head_dtype_policy_preserves_master_bits(numeric, datetime):
    m = model(numeric=numeric, datetime=datetime, dtype=torch.float32)
    with torch.no_grad():
        for head in m.dec_dict.values():
            head.weight.fill_(1.0012345)
            head.bias.fill_(0.000123456789)
    original = {k: v.clone() for k, v in m.state_dict().items()}
    ids = {k: id(v) for k, v in m.named_parameters()}
    m.set_compute_dtype()
    m.set_compute_dtype()
    for name, p in m.named_parameters():
        protected = (name.startswith("dec_dict.number.") and numeric == "bits") or (
            name.startswith("dec_dict.datetime.") and datetime == "calendar"
        )
        dtype = torch.float32 if protected else torch.bfloat16
        assert id(p) == ids[name]
        assert p.dtype == dtype
        assert torch.equal(p, original[name].to(dtype))


@pytest.mark.parametrize("datetime", ["scalar", "calendar"])
@pytest.mark.parametrize("key", ["number_features", "number_target_values"])
@pytest.mark.parametrize("problem", ["missing", "dtype", "shape"])
def test_bits_requires_exact_numeric_interface(datetime, key, problem):
    b = batch()
    if problem == "missing":
        del b[key]
    elif problem == "dtype":
        b[key] = b[key].to(
            torch.float32 if key == "number_features" else torch.bfloat16
        )
    else:
        b[key] = b[key][..., 0]
    with pytest.raises(ValueError, match=key):
        model(datetime=datetime)(b, return_embeddings=False)


def test_numeric_fields_sort_and_scalar_datetime_needs_no_sidecar():
    m = model()
    b = {k: v for k, v in batch().items() if not k.startswith("datetime_")}
    b["datetime_values"] = datetime_batch()["datetime_values"]
    saved = {k: v.clone() for k, v in b.items()}
    order = torch.tensor([[1, 3, 2, 0]])
    aligned = m._numeric_values(b, order)
    for key, value in aligned.items():
        assert torch.equal(value, b[key][:, [1, 3, 2, 0]])
    loss, _, per_type, mask = m(b, return_embeddings=False)
    assert mask.tolist() == [[False, False, True, True]]
    expected_number = b["number_target_values"][0, 2, 0] - 0.5
    assert torch.equal(per_type["number"], expected_number)
    assert loss.dtype == torch.float32
    for key, value in b.items():
        assert torch.equal(value, saved[key])


@pytest.mark.parametrize("datetime", ["scalar", "calendar"])
def test_masked_numeric_payload_cannot_leak_into_predictions(datetime):
    torch.manual_seed(0)
    m = model(datetime=datetime, num_blocks=1)
    with torch.no_grad():
        for head in m.dec_dict.values():
            head.weight.copy_(torch.tensor([1, -2, 3, 4]) / 32)
        m.blocks[0].attns["feat"].wv.weight.copy_(torch.eye(4))
        m.blocks[0].attns["feat"].wo.weight.copy_(torch.eye(4))
    b = batch()
    b["node_idxs"].fill_(1)
    initial_embeddings = m(b, return_embeddings=True)
    original = m(b, return_embeddings=False)
    b["number_features"][0, 2].fill_(float("nan"))
    b["number_target_values"][0, 2, 0] = 3.001234
    b["number_target_values"][0, [0, 1, 3]] = float("nan")
    # The old scalar payload is entirely irrelevant in bits mode.
    b["number_values"].fill_(float("nan"))
    assert torch.equal(initial_embeddings, m(b, return_embeddings=True))
    changed = m(b, return_embeddings=False)
    for name in net.SEM_TYPE_NAMES:
        assert torch.equal(original[1][name], changed[1][name])
    assert changed[0] > original[0]
    # Positive control: actual visible number context must be usable.
    b["number_features"][0, 1] = features(torch.tensor(10.12345))
    context_changed = m(b, return_embeddings=False)
    assert not torch.equal(changed[1]["number"], context_changed[1]["number"])


@pytest.mark.parametrize("datetime", ["scalar", "calendar"])
@pytest.mark.parametrize("loss_fn", ["huber", "l1", "bce"])
def test_hidden_nonfinite_payload_is_safe_in_forward_and_backward(datetime, loss_fn):
    m = model(datetime=datetime, num_blocks=1, loss_fn=loss_fn)
    b = batch()
    b["is_padding"][0, 3] = True
    b["is_targets"][0, 3] = True
    b["col_name_values"][0, 3] = float("nan")
    b["number_features"][0, [0, 2, 3]] = float("nan")
    b["number_target_values"][0, [0, 1, 3]] = float("inf")
    b["number_values"].fill_(float("nan"))
    b["text_values"].fill_(float("nan"))
    b["datetime_values"][0, 1:] = float("nan")
    b["datetime_target_values"][0, 1:] = float("nan")
    b["datetime_features"].fill_(float("nan"))  # Only date target, never context.
    loss, predictions, per_type, mask = m(b, return_embeddings=False)
    assert mask.sum() == 2
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert all(
        t.dtype == torch.float32 and torch.isfinite(t) for t in per_type.values()
    )
    assert all(torch.isfinite(t).all() for t in predictions.values())
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


@pytest.mark.parametrize("datetime", ["scalar", "calendar"])
def test_valid_zero_target_padding_and_empty_sequences(datetime):
    m = model(datetime=datetime)
    b = batch()
    b["is_targets"].zero_()
    b["is_targets"][0, [2, 3]] = True
    b["is_padding"][0, 3] = True
    b["number_target_values"][0, 2, 0] = 0.0
    b["number_features"][0, 2].zero_()
    with torch.no_grad():
        m.dec_dict["number"].bias.fill_(1.0)
    loss, _, per_type, mask = m(b, return_embeddings=False)
    assert mask.sum() == 1
    assert loss.item() == per_type["number"].item() == 0.5
    b["is_targets"].zero_()
    empty, _, _, mask = m(b, return_embeddings=False)
    assert empty.item() == 0 and not mask.any()
    empty.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )


@pytest.mark.parametrize("datetime", ["scalar", "calendar"])
def test_number_head_and_labels_retain_off_bf16_grid_supervision(datetime):
    m = model(datetime=datetime)
    b = batch()
    b["is_targets"][0, 0] = False
    b["number_target_values"][0, 2, 0] = 1.001
    with torch.no_grad():
        m.dec_dict["number"].bias.fill_(1.00390625)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, predictions, _, _ = m(b, return_embeddings=False)
    pred = predictions["number"]
    assert pred.dtype == torch.float32
    assert not torch.equal(pred, pred.bfloat16().float())
    assert predictions["datetime"].dtype == (
        torch.float32 if datetime == "calendar" else torch.bfloat16
    )
    residual = float(pred[0, 0, 0].detach()) - float(b["number_target_values"][0, 2, 0])
    assert loss.item() == pytest.approx(residual**2 / 2, rel=1e-6)
    loss.backward()
    head = m.dec_dict["number"]
    assert head.weight.grad.dtype == head.bias.grad.dtype == torch.float32
    assert head.bias.grad.item() == pytest.approx(residual, abs=1e-8)
    assert m.enc_dict["number"].weight.grad.dtype == torch.bfloat16


@pytest.mark.parametrize("datetime", ["scalar", "calendar"])
@pytest.mark.parametrize("part", ["weight", "bias"])
def test_bits_rejects_narrowed_live_numeric_head(datetime, part):
    m = model(datetime=datetime)
    p = getattr(m.dec_dict["number"], part)
    p.data = p.data.bfloat16()
    with pytest.raises(ValueError, match="number head must remain FP32"):
        m(batch(), return_embeddings=False)


def test_bounded_context_extrema_and_one_ulp_controlled_projection():
    m = model(d_model=8)
    b = batch()
    raw = torch.tensor(
        [0x3F800000, 0x3F800001, 0x7F7FFFFE, 0x7F7FFFFF], dtype=torch.int32
    ).view(torch.float32)
    b["sem_types"].zero_()
    b["is_targets"].zero_()
    b["col_name_idxs"].zero_()
    b["number_features"][0] = features(raw)
    b["number_target_values"][0, :, 0] = raw
    b["number_values"][0, :, 0] = raw.bfloat16()
    with torch.no_grad():
        m.enc_dict["number"].weight.zero_()
        m.enc_dict["number"].weight[:5].copy_(torch.eye(5))
        m.enc_dict["number"].bias.zero_()
        m.enc_dict["col_name"].weight.zero_()
        m.enc_dict["col_name"].bias.fill_(0.25)
    linear = m.enc_dict["number"](b["number_features"])
    norm = m.norm_dict["number"](linear)
    col = m.norm_dict["col_name"](m.enc_dict["col_name"](b["col_name_values"]))
    for stage in [linear, norm, norm + col, m(b, return_embeddings=True)]:
        assert torch.isfinite(stage).all()
        assert stage.dtype == torch.bfloat16
        assert not torch.equal(stage[0, 0], stage[0, 1])
        assert not torch.equal(stage[0, 2], stage[0, 3])
    # This checks bounded context only, not finite losses for extreme targets.


@pytest.mark.parametrize("datetime", ["scalar", "calendar"])
@pytest.mark.parametrize("loss_fn", ["huber", "l1", "bce"])
def test_precision_loss_dispatch_counts_and_gradients_match_double(datetime, loss_fn):
    m = model(datetime=datetime, loss_fn=loss_fn)
    with torch.no_grad():
        for head in m.dec_dict.values():
            head.weight.fill_(0.1001234)
            head.bias.fill_(0.010123)
    x = ((torch.arange(32).reshape(2, 4, 4) % 11) - 5).bfloat16() / 4
    x.requires_grad_()
    sem_types = torch.tensor([[0, 0, 2, 1], [0, 2, 1, 0]])
    targets = torch.ones((2, 4), dtype=torch.bool)
    padding = torch.tensor([[False] * 4, [False, False, False, True]])
    valid = torch.tensor([[False, False, True, False], [False] * 4])
    labels = {
        "number": torch.tensor(
            [
                [[1.00123], [-0.23567], [float("nan")], [float("nan")]],
                [[0.0], [float("nan")], [float("nan")], [float("nan")]],
            ]
        ),
        "datetime": torch.tensor(
            [[[0.0], [0.0], [2.34567], [0.0]], [[0.0], [-1.23456], [0.0], [0.0]]],
            dtype=torch.float32 if datetime == "calendar" else torch.bfloat16,
        ),
        "text": torch.full((2, 4, 2), -0.23456, dtype=torch.bfloat16),
    }
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, predictions, per_type, mask = m._precision_loss(
            x,
            labels,
            sem_types,
            targets,
            padding,
            valid if datetime == "calendar" else None,
        )
    expected_mask = ~padding
    if datetime == "calendar":
        expected_mask &= (sem_types != 2) | valid
    assert torch.equal(mask, expected_mask)
    head = m.dec_dict["number"]
    w = head.weight.detach().double().requires_grad_()
    bias = head.bias.detach().double().requires_grad_()
    number = F.linear(x.detach().double(), w, bias)
    torch.testing.assert_close(
        predictions["number"].double(), number, atol=1e-7, rtol=1e-6
    )
    cell_losses = torch.zeros((2, 4), dtype=torch.float64)
    for index, name in enumerate(net.SEM_TYPE_NAMES):
        chosen = expected_mask & (sem_types == index)
        pred = number if name == "number" else predictions[name].detach().double()
        pred, target = pred[chosen], labels[name].double()[chosen]
        residual = (pred - target).abs()
        if loss_fn == "l1":
            value = residual.mean(-1)
        elif loss_fn == "huber" or (name == "datetime" and datetime == "calendar"):
            value = torch.where(
                residual < 1, residual.square() / 2, residual - 0.5
            ).mean(-1)
        else:
            weight = (target != 0).double()
            value = (
                F.binary_cross_entropy_with_logits(
                    pred, (target > 0).double(), reduction="none"
                )
                * weight
            ).sum(-1) / weight.sum(-1).clamp_min(1)
        cell_losses[chosen] = value
        torch.testing.assert_close(
            per_type[name].double(), value.mean(), atol=2e-7, rtol=1e-6
        )
    expected = (cell_losses.sum(1) / expected_mask.sum(1)).mean()
    torch.testing.assert_close(loss.double(), expected, atol=2e-7, rtol=1e-6)
    gradients = torch.autograd.grad(expected, (w, bias))
    loss.backward()
    for p, reference in zip((head.weight, head.bias), gradients, strict=True):
        assert p.grad.dtype == torch.float32
        torch.testing.assert_close(p.grad.double(), reference, atol=2e-7, rtol=1e-6)
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("datetime", ["scalar", "calendar"])
def test_numeric_forward_captures_single_cpu_graph(datetime):
    m, b = model(datetime=datetime), batch()
    expected = m(b, return_embeddings=False)
    compiled = torch.compile(m, backend="eager", fullgraph=True)
    assert_outputs_equal(expected, compiled(b, return_embeddings=False))


@pytest.mark.parametrize("accepted", [False, True])
def test_numeric_cuda_setup_uses_shared_guard_before_cast(monkeypatch, accepted):
    events = []

    def validate():
        events.append("validate")
        if not accepted:
            raise ValueError("unsafe policy")

    m = SimpleNamespace(
        datetime_encoding="scalar",
        numeric_encoding="bits",
        dec_dict={
            "number": SimpleNamespace(
                weight=SimpleNamespace(device=torch.device("cuda"))
            )
        },
        named_children=lambda: [
            ("body", SimpleNamespace(to=lambda **kw: events.append("cast")))
        ],
    )
    monkeypatch.setattr(net, "validate_cuda_matmul_precision", validate)
    if accepted:
        assert net.RelationalTransformer.set_compute_dtype(m) is m
        assert events == ["validate", "cast"]
    else:
        with pytest.raises(ValueError, match="unsafe policy"):
            net.RelationalTransformer.set_compute_dtype(m)
        assert events == ["validate"]


@pytest.mark.parametrize("numeric", ["scalar", "bits"])
@pytest.mark.parametrize("datetime", ["scalar", "calendar"])
@pytest.mark.parametrize("tag_source", ["config", "file"])
def test_independent_checkpoint_roundtrip(tmp_path, numeric, datetime, tag_source):
    m = model(numeric=numeric, datetime=datetime)
    with torch.no_grad():
        for head in m.dec_dict.values():
            head.weight.fill_(0.1001234)
            head.bias.fill_(0.010123)
    modes = {"numeric_encoding": numeric, "datetime_encoding": datetime}
    config = {"model": {**DIMS, **(modes if tag_source == "config" else {})}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    path = tmp_path / "model.safetensors"
    checkpoints.save_model(m.state_dict(), path, modes if tag_source == "file" else {})
    loaded = net.RelationalTransformer.from_pretrained(tmp_path).set_compute_dtype()
    assert loaded.numeric_encoding == numeric and loaded.datetime_encoding == datetime
    for name, value in m.state_dict().items():
        assert loaded.state_dict()[name].dtype == value.dtype
        assert torch.equal(loaded.state_dict()[name], value)
    assert_outputs_equal(
        m(batch(), return_embeddings=False), loaded(batch(), return_embeddings=False)
    )


def test_standalone_numeric_file_tag_and_unmarked_legacy(tmp_path):
    path = tmp_path / "model.safetensors"
    m = model()
    checkpoints.save_model(m.state_dict(), path, {"numeric_encoding": "bits"})
    loaded = net.RelationalTransformer.from_pretrained(path, **DIMS).set_compute_dtype()
    assert loaded.numeric_encoding == "bits" and loaded.datetime_encoding == "scalar"
    checkpoints.save_model(model(numeric="scalar").state_dict(), path)
    loaded = net.RelationalTransformer.from_pretrained(path, **DIMS).set_compute_dtype()
    assert loaded.numeric_encoding == loaded.datetime_encoding == "scalar"
    assert all(p.dtype == torch.bfloat16 for p in loaded.parameters())


@pytest.mark.parametrize(
    "config_mode,file_mode", [("scalar", "bits"), ("bits", "scalar")]
)
def test_numeric_checkpoint_conflicts_reject(tmp_path, config_mode, file_mode):
    path = tmp_path / "model.safetensors"
    (tmp_path / "config.json").write_text(
        json.dumps({"model": {**DIMS, "numeric_encoding": config_mode}})
    )
    checkpoints.save_model(
        model(numeric=file_mode).state_dict(), path, {"numeric_encoding": file_mode}
    )
    with pytest.raises(ValueError, match="conflicting numeric_encoding"):
        net.RelationalTransformer.from_pretrained(path)
    with pytest.raises(ValueError, match="conflicting numeric_encoding"):
        checkpoints.load_model(path, numeric_encoding=config_mode)


@pytest.mark.parametrize("tag_source", ["config", "file"])
def test_numeric_unknown_encoding_rejected(tmp_path, tag_source):
    path = tmp_path / "model.safetensors"
    config = {"model": DIMS.copy()}
    metadata = {}
    if tag_source == "config":
        config["model"]["numeric_encoding"] = "other"
    else:
        metadata["numeric_encoding"] = "other"
    (tmp_path / "config.json").write_text(json.dumps(config))
    checkpoints.save_model(model().state_dict(), path, metadata)
    with pytest.raises(ValueError, match="unknown numeric_encoding"):
        net.RelationalTransformer.from_pretrained(path)
    with pytest.raises(ValueError, match="unknown numeric_encoding"):
        model(numeric="other")


@pytest.mark.parametrize("tag_source", ["config", "file"])
@pytest.mark.parametrize("part", ["weight", "bias"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_numeric_load_and_warmstart_reject_narrowed_source_head(
    tmp_path, tag_source, part, dtype
):
    state = model(datetime="calendar").state_dict()
    key = "dec_dict.number." + part
    state[key] = state[key].to(dtype)
    path = tmp_path / "model.safetensors"
    modes = {"datetime_encoding": "calendar", "numeric_encoding": "bits"}
    (tmp_path / "config.json").write_text(
        json.dumps({"model": {**DIMS, **(modes if tag_source == "config" else {})}})
    )
    checkpoints.save_model(state, path, modes if tag_source == "file" else {})
    with pytest.raises(ValueError, match="stored FP32 " + key):
        checkpoints.validate_precision_state_dict(state, **modes)
    with pytest.raises(ValueError, match="stored FP32 " + key):
        checkpoints.load_model(path, **modes)
    with pytest.raises(ValueError, match="stored FP32 " + key):
        net.RelationalTransformer.from_pretrained(path)
    assert state[key].dtype == dtype


def test_precision_validator_checks_only_active_heads_and_requires_keys():
    scalar = model(numeric="scalar").state_dict()
    checkpoints.validate_precision_state_dict(scalar)
    # Explicit transfer to a precise number head cannot claim widened BF16 weights.
    with pytest.raises(ValueError, match="stored FP32 dec_dict.number"):
        checkpoints.validate_precision_state_dict(scalar, numeric_encoding="bits")
    bits = model().state_dict()
    checkpoints.validate_precision_state_dict(bits, numeric_encoding="bits")
    assert bits["dec_dict.datetime.weight"].dtype == torch.bfloat16
    del bits["dec_dict.number.bias"]
    with pytest.raises(ValueError, match="missing dec_dict.number.bias"):
        checkpoints.validate_precision_state_dict(bits, numeric_encoding="bits")


def test_unmarked_shape_does_not_infer_numeric_encoding_or_transfer(tmp_path):
    path = tmp_path / "model.safetensors"
    checkpoints.save_model(model().state_dict(), path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        net.RelationalTransformer.from_pretrained(path, **DIMS)
    with pytest.raises(ValueError, match="explicit warm-start"):
        net.RelationalTransformer.from_pretrained(path, **DIMS, numeric_encoding="bits")
