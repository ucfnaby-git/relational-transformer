"""CPU contracts for exact calendar inputs and approximate FP32 scalar targets.

Attention uses the dense CPU SDPA reference; these tests do not validate compiled
CUDA FlexAttention or claim that arbitrary learned embeddings are injective.
"""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

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


def model(encoding="calendar", dtype=torch.bfloat16, **kwargs):
    dims = {**DIMS, **kwargs}
    return net.RelationalTransformer(
        **dims,
        compile=False,
        materialize_attn_masks=True,
        datetime_encoding=encoding,
    ).set_compute_dtype(dtype)


def batch(dtype=torch.bfloat16):
    # Unsorted columns, datetime and numeric targets, numeric/text context.
    return {
        "node_idxs": torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
        "f2p_nbr_idxs": torch.full((1, 4, 5), -1, dtype=torch.int32),
        "col_name_idxs": torch.tensor([[4, 1, 3, 2]], dtype=torch.int32),
        "table_name_idxs": torch.zeros((1, 4), dtype=torch.int32),
        "is_padding": torch.zeros((1, 4), dtype=torch.bool),
        "sem_types": torch.tensor([[2, 0, 0, 1]], dtype=torch.int32),
        "is_targets": torch.tensor([[True, False, True, False]]),
        "col_name_values": torch.tensor(
            [[[0.25, -0.5], [0.5, 0.75], [-0.5, 1], [0.25, 1.25]]], dtype=dtype
        ),
        "number_values": torch.tensor([[[0.5], [-0.25], [1.25], [0]]], dtype=dtype),
        "datetime_values": torch.tensor([[[1.25], [0], [0], [0]]], dtype=dtype),
        "text_values": torch.tensor(
            [[[0, 0], [0, 0], [0, 0], [0.5, -0.75]]], dtype=dtype
        ),
        "datetime_features": torch.zeros((1, 4, 14), dtype=torch.bfloat16),
        "datetime_target_values": torch.tensor([[[1.25], [0], [0], [0]]]),
        "datetime_seconds": torch.tensor([[1_800_000_000, 0, 0, 0]], dtype=torch.int64),
        "datetime_nanoseconds": torch.tensor(
            [[123_456_789, 0, 0, 0]], dtype=torch.int32
        ),
        "datetime_valid": torch.tensor([[True, False, False, False]]),
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("loss_fn", ["huber", "l1", "bce"])
def test_legacy_outputs_match_unmodified_wheel(dtype, loss_fn):
    # Golden values captured from cf91f45's unmodified isolated wheel on CPU.
    # Explicit dyadic parameters avoid reliance on random initialization.
    m = model("scalar", dtype=dtype, loss_fn=loss_fn)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(((torch.arange(p.numel()).reshape(p.shape) % 13) - 6).float() / 8)
    loss, outputs, _, targets = m(batch(dtype), return_embeddings=False)
    losses = {
        torch.float32: {
            "huber": 0.7319756746292114,
            "l1": 1.2319756746292114,
            "bce": 0.6852219104766846,
        },
        torch.bfloat16: {"huber": 0.73828125, "l1": 1.234375, "bce": 0.6875},
    }
    predictions = {
        torch.float32: [
            0.7296363115310669,
            0.7324796915054321,
            -0.07348459959030151,
            0.10953325033187866,
        ],
        torch.bfloat16: [0.73046875, 0.73046875, -0.08349609375, 0.111328125],
    }
    assert loss.item() == pytest.approx(losses[dtype][loss_fn], abs=1e-7, rel=0)
    expected = torch.tensor(predictions[dtype]).reshape(1, 4, 1)
    torch.testing.assert_close(outputs["number"].float(), expected, atol=1e-7, rtol=0)
    torch.testing.assert_close(outputs["datetime"].float(), expected, atol=1e-7, rtol=0)
    assert targets.tolist() == [[False, False, True, True]]
    assert all(p.dtype == dtype for p in m.parameters())


def test_constructor_default_is_legacy_scalar():
    m = net.RelationalTransformer(**DIMS, compile=False, materialize_attn_masks=True)
    assert m.datetime_encoding == "scalar"
    assert m.enc_dict["datetime"].in_features == 1
    m.set_compute_dtype()
    assert m.dec_dict["datetime"].weight.dtype == torch.bfloat16


def test_dtype_setup_preserves_head_bits_and_parameter_identity():
    m = model(dtype=torch.float32, num_blocks=1)
    head = m.dec_dict["datetime"]
    with torch.no_grad():
        head.weight.fill_(1.001)
        head.bias.fill_(0.000123456789)
    original = {k: v.clone() for k, v in head.state_dict().items()}
    identities = {k: id(v) for k, v in m.named_parameters()}
    assert m.set_compute_dtype() is m
    m.set_compute_dtype()
    for name, p in m.named_parameters():
        assert id(p) == identities[name]
        expected = (
            torch.float32 if name.startswith("dec_dict.datetime.") else torch.bfloat16
        )
        assert p.dtype == expected
    for name, p in head.state_dict().items():
        assert torch.equal(p, original[name])
    assert not torch.equal(head.weight, head.weight.bfloat16().float())


@pytest.mark.parametrize(
    "key",
    [
        "datetime_features",
        "datetime_target_values",
        "datetime_seconds",
        "datetime_nanoseconds",
        "datetime_valid",
    ],
)
def test_calendar_requires_exact_native_fields(key):
    b = batch()
    del b[key]
    with pytest.raises(ValueError, match="exact native batch.*" + key):
        model()(b, return_embeddings=True)


@pytest.mark.parametrize(
    "key",
    [
        "datetime_features",
        "datetime_target_values",
        "datetime_seconds",
        "datetime_nanoseconds",
        "datetime_valid",
    ],
)
def test_calendar_rejects_wrong_dtypes(key):
    b = batch()
    b[key] = b[key].double()
    with pytest.raises(ValueError, match=key + " must have shape"):
        model()(b, return_embeddings=True)


def test_calendar_rejects_broadcastable_scalar_targets():
    b = batch()
    b["datetime_target_values"] = b["datetime_target_values"][..., 0]
    with pytest.raises(ValueError, match="datetime_target_values must have shape"):
        model()(b, return_embeddings=False)


def test_calendar_fields_follow_sort_and_leave_source_unchanged():
    m = model()
    b = batch()
    b["datetime_features"][0] = torch.arange(56).reshape(4, 14).bfloat16() / 256
    b["datetime_seconds"][0] = torch.tensor([10, 20, 30, 40])
    b["datetime_nanoseconds"][0] = torch.tensor([1, 2, 3, 4])
    saved = {k: v.clone() for k, v in b.items()}
    order = torch.tensor([[1, 3, 2, 0]])
    aligned = m._calendar_values(b, order)
    for key, value in aligned.items():
        assert torch.equal(value, b[key][:, [1, 3, 2, 0]])
        assert torch.equal(b[key], saved[key])
    _, _, losses, targets = m(b, return_embeddings=False)
    assert targets.tolist() == [[False, False, True, True]]
    assert losses["datetime"].item() == 0.75  # Huber(0, 1.25), without BF16 labels.


def test_controlled_calendar_projection_norm_and_column_sum_keep_date_pair_distinct():
    m = model(d_model=16)
    b = batch()
    b["sem_types"].fill_(2)
    b["is_targets"].fill_(False)
    b["datetime_valid"].fill_(True)
    b["col_name_idxs"].zero_()
    # One day apart, identical columns. Calendar fields are integers / 256.
    a = torch.tensor([0, 7, 233, 7, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0]) / 256
    c = a.clone()
    c[4] = 13 / 256
    b["datetime_features"][0] = torch.stack([a, c, a, c]).bfloat16()
    assert torch.equal(b["datetime_features"][0, 0].float(), a)
    with torch.no_grad():
        m.enc_dict["datetime"].weight.zero_()
        m.enc_dict["datetime"].weight[:14, :14].copy_(torch.eye(14))
        m.enc_dict["datetime"].bias.zero_()
        m.enc_dict["col_name"].weight.zero_()
        m.enc_dict["col_name"].bias.fill_(0.25)
    features = net._calendar_context_features(
        b["datetime_features"], b["datetime_target_values"], b["datetime_valid"]
    )
    projected = m.enc_dict["datetime"](features)
    normalized = m.norm_dict["datetime"](projected)
    column = m.norm_dict["col_name"](m.enc_dict["col_name"](b["col_name_values"]))
    for stage in [
        projected,
        normalized,
        normalized + column,
        m(b, return_embeddings=True),
    ]:
        assert stage.dtype == torch.bfloat16
        assert not torch.equal(stage[0, 0], stage[0, 1])
    # This is a controlled sensitivity test, not arbitrary-weight invertibility.


def test_hidden_datetime_payload_never_changes_predictions():
    torch.manual_seed(0)
    m = model(num_blocks=1)
    with torch.no_grad():
        for head in m.dec_dict.values():
            head.weight.copy_(torch.tensor([1, -2, 3, 4]) / 32)
        # Activate cross-cell mixing; residual outputs are zero at initialization.
        m.blocks[0].attns["feat"].wv.weight.copy_(torch.eye(4))
        m.blocks[0].attns["feat"].wo.weight.copy_(torch.eye(4))
    b = batch()
    b["node_idxs"].fill_(1)  # Every cell attends to every cell via feature attention.
    b["sem_types"][0, 1] = 2
    b["datetime_valid"][0, 1] = True
    b["datetime_features"][0, 1] = torch.arange(14).bfloat16() / 256
    initial_embeddings = m(b, return_embeddings=True)
    loss, outputs, _, _ = m(b, return_embeddings=False)
    b["datetime_features"][0, 0].fill_(float("nan"))
    b["datetime_seconds"][0, 0] = -1
    b["datetime_nanoseconds"][0, 0] = 999999999
    b["datetime_target_values"][0, 0, 0] = 1.5
    assert torch.equal(initial_embeddings, m(b, return_embeddings=True))
    changed_loss, changed_outputs, _, _ = m(b, return_embeddings=False)
    for t in outputs:
        assert torch.equal(outputs[t], changed_outputs[t])
    assert changed_loss > loss
    # Positive control: a visible datetime context is allowed to affect the target.
    b["datetime_features"][0, 1] = torch.arange(13, -1, -1).bfloat16() / 16
    _, context_outputs, _, _ = m(b, return_embeddings=False)
    assert not torch.equal(
        outputs["datetime"][:, -1], context_outputs["datetime"][:, -1]
    )


def _normalized_bit_patterns():
    corners = torch.tensor(
        [
            0,
            -2147483648,
            1,
            -2147483647,
            0x007FFFFF,
            0x00800000,
            0x3F800000,
            0x3F800001,
            0x3F810000,
            0x7F7FFFFF,
            -8388609,
            0x7FC00000,
            0x7F800000,
            -8388608,
        ],
        dtype=torch.int32,
    )
    random = torch.randint(
        -2147483648,
        2147483647,
        (1024,),
        generator=torch.Generator().manual_seed(89),
        dtype=torch.int32,
    )
    return torch.cat([corners, random]).reshape(1, -1, 1)


@pytest.mark.parametrize("autocast", [False, True])
def test_calendar19_preserves_normalized_fp32_bits_and_native14(autocast):
    bits = _normalized_bit_patterns()
    values = bits.view(torch.float32)
    permitted = torch.isfinite(values[..., 0])
    calendar = torch.arange(14).bfloat16().expand(1, values.shape[1], 14) / 256
    before = values.view(torch.int32).clone()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        features = net._calendar_context_features(calendar, values, permitted)
    assert features.shape == (*permitted.shape, 19)
    assert features.dtype == torch.bfloat16
    assert torch.isfinite(features).all()
    assert torch.equal(features[..., :14][permitted], calendar[permitted])
    assert (features[~permitted] == 0).all()
    # Reconstruct bytes independently with integer place values.
    digits = (features[..., 15:].float() * 256).long()
    reconstructed = (digits * torch.tensor([1 << 24, 1 << 16, 1 << 8, 1])).sum(-1)
    expected = torch.where(permitted, bits[..., 0], 0)
    assert torch.equal(reconstructed.int(), expected)
    assert torch.equal(values.view(torch.int32), before)
    exponent_zero = (expected & 0x7F800000) == 0
    assert (features[..., 14].contiguous().view(torch.int16)[exponent_zero] == 0).all()
    assert features[..., 14].abs().max() <= 1


def test_normalized_byte_channels_keep_distinction_lost_by_coarse_cue():
    values = torch.tensor([1.0, 129 / 128]).reshape(1, 2, 1)
    assert values.bfloat16()[0, 0] != values.bfloat16()[0, 1]
    features = net._calendar_context_features(
        torch.zeros(1, 2, 14, dtype=torch.bfloat16),
        values,
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert features[0, :, 14].tolist() == [0.5, 0.5]
    assert features[0, 0, 15:].tolist() == [63 / 256, 128 / 256, 0, 0]
    assert features[0, 1, 15:].tolist() == [63 / 256, 129 / 256, 0, 0]


def test_calendar19_masks_before_arithmetic_and_bit_extraction():
    values = torch.tensor([1.25, float("nan"), float("inf"), -float("inf")]).reshape(
        1, 4, 1
    )
    calendar = torch.full((1, 4, 14), float("nan"), dtype=torch.bfloat16)
    calendar[:, 0] = torch.arange(14).bfloat16() / 256
    permitted = torch.tensor([[True, False, False, False]])
    observed = net._calendar_context_features(calendar, values, permitted)
    assert torch.isfinite(observed).all()
    assert (observed[:, 1:] == 0).all()
    values[:, 1:] = -12345.0
    calendar[:, 1:] = 123.0
    assert torch.equal(
        observed, net._calendar_context_features(calendar, values, permitted)
    )


def test_calendar19_feature_helper_cpu_inductor_matches_eager():
    values = _normalized_bit_patterns().view(torch.float32)
    permitted = torch.isfinite(values[..., 0])
    calendar = torch.zeros(1, values.shape[1], 14, dtype=torch.bfloat16)
    expected = net._calendar_context_features(calendar, values, permitted)
    compiled = torch.compile(net._calendar_context_features, fullgraph=True)
    assert torch.equal(compiled(calendar, values, permitted), expected)
    assert not torch.cuda.is_initialized()


def test_same_raw_calendar_different_training_normalizations_reach_model():
    # Native witness: events times [10,11], unreachable calibration table [0,1]
    # or [30,31], with both date columns repeating each timestamp. The same
    # masked raw date is recoverable by copying its visible same-row date.
    torch.manual_seed(17)
    m = model(d_model=32, num_blocks=1)
    b = batch()
    b["node_idxs"].fill_(1)
    b["sem_types"].fill_(2)
    b["is_targets"][:] = torch.tensor([[True, False, False, False]])
    b["is_padding"][0, 3] = True
    b["datetime_valid"][:] = ~b["is_padding"]
    raw = torch.tensor([10.0, 10.0, 11.0, 0.0]).reshape(1, 4, 1)
    date = torch.tensor([0, 7, 208, 1, 1, 0, 0, 10, 0, 0, 0, 0, 0, 0]) / 256
    b["datetime_features"][:] = date.bfloat16()
    b["datetime_features"][0, 2, 7] = 11 / 256
    b["datetime_seconds"][:] = 946684800 + raw[..., 0].long()
    b["datetime_nanoseconds"].zero_()
    a = {k: v.clone() for k, v in b.items()}
    a["datetime_target_values"] = ((raw.double() - 5.5) / (25.25**0.5)).float()
    b["datetime_target_values"] = ((raw.double() - 20.5) / (100.25**0.5)).float()
    assert a["datetime_target_values"][0, 0, 0].item() == pytest.approx(0.8955334)
    assert b["datetime_target_values"][0, 0, 0].item() == pytest.approx(-1.0486900)
    assert torch.equal(a["datetime_features"], b["datetime_features"])
    with torch.no_grad():
        m.enc_dict["datetime"].weight.zero_()
        m.enc_dict["datetime"].weight[:19, :19].copy_(torch.eye(19))
        m.enc_dict["datetime"].bias.zero_()
        # Ensure the visible context actually reaches the masked target.
        attn = m.blocks[0].attns["feat"]
        attn.wq.weight.zero_()
        attn.wk.weight.zero_()
        attn.wv.weight.copy_(torch.eye(32))
        attn.wo.weight.copy_(torch.eye(32))
    inputs = []
    handle = m.enc_dict["datetime"].register_forward_pre_hook(
        lambda _module, args: inputs.append(args[0].detach().clone())
    )
    hidden_a = m(a, return_embeddings=True)
    hidden_b = m(b, return_embeddings=True)
    handle.remove()
    order = torch.tensor([[1, 2, 0, 3]])
    target = a["is_targets"].gather(1, order)
    assert inputs[0].shape[-1] == m.enc_dict["datetime"].in_features == 19
    assert torch.equal(inputs[0][..., :14], inputs[1][..., :14])
    assert torch.equal(
        inputs[0][..., 14:],
        net._calendar_context_features(
            a["datetime_features"].gather(1, order[..., None].expand(-1, -1, 14)),
            a["datetime_target_values"].gather(1, order[..., None]),
            (~a["is_targets"] & ~a["is_padding"]).gather(1, order),
        )[..., 14:],
    )
    assert (inputs[0][target] == 0).all() and (inputs[1][target] == 0).all()
    assert not torch.equal(hidden_a[target], hidden_b[target])


@pytest.mark.parametrize("loss_fn", ["huber", "l1", "bce"])
def test_irrelevant_nan_operands_are_safe_in_forward_and_backward(loss_fn):
    m = model(num_blocks=1, loss_fn=loss_fn)
    b = batch()
    b["is_padding"][0, 3] = True
    b["is_targets"][0, 3] = True  # Padding must not affect count or loss.
    b["datetime_features"].fill_(float("nan"))
    b["datetime_target_values"][0, 1:] = float("nan")
    b["number_values"][0, [0, 3]] = float("nan")
    b["text_values"].fill_(float("nan"))
    b["col_name_values"][0, 3] = float("nan")
    loss, outputs, per_type, targets = m(b, return_embeddings=False)
    assert targets.sum().item() == 2
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert all(
        v.dtype == torch.float32 and torch.isfinite(v) for v in per_type.values()
    )
    assert all(torch.isfinite(v).all() for v in outputs.values())
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_invalid_datetime_and_padding_do_not_dilute_loss():
    m = model()
    b = batch()
    b["datetime_valid"][0, 0] = False
    b["datetime_target_values"][0, 0] = float("nan")
    b["datetime_features"][0, 0] = float("nan")
    b["is_padding"][0, 3] = True
    b["is_targets"][0, 3] = True
    loss, _, types, targets = m(b, return_embeddings=False)
    assert targets.sum().item() == 1
    assert loss.item() == types["number"].item() == 0.75
    assert types["datetime"].item() == 0
    b["is_targets"].zero_()
    empty_loss, _, _, targets = m(b, return_embeddings=False)
    assert empty_loss.item() == 0
    assert targets.sum().item() == 0
    empty_loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )


@pytest.mark.parametrize(
    "loss_fn, expected", [("huber", 1.5), ("l1", 2.0), ("bce", 1.5)]
)
def test_datetime_remains_regression_under_numeric_bce(loss_fn, expected):
    m = model(loss_fn=loss_fn)
    b = batch()
    b["is_targets"][0, 2] = False
    b["datetime_target_values"][0, 0, 0] = 2.0
    loss, _, types, _ = m(b, return_embeddings=False)
    assert loss.item() == types["datetime"].item() == expected


def test_datetime_fp32_head_and_loss_survive_autocast():
    m = model()
    b = batch()
    b["is_targets"][0, 2] = False
    b["datetime_target_values"][0, 0, 0] = 1.001
    with torch.no_grad():
        m.dec_dict["datetime"].bias.fill_(1.00390625)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, outputs, _, _ = m(b, return_embeddings=False)
    pred = outputs["datetime"]
    assert pred.dtype == torch.float32
    assert outputs["number"].dtype == torch.bfloat16
    assert not torch.equal(pred, pred.bfloat16().float())
    residual = float(pred[0, 0, 0].detach()) - float(
        b["datetime_target_values"][0, 0, 0]
    )
    assert loss.item() == pytest.approx(0.5 * residual**2, rel=1e-6)
    loss.backward()
    assert m.dec_dict["datetime"].weight.grad.dtype == torch.float32
    assert m.dec_dict["datetime"].bias.grad.item() == pytest.approx(residual, abs=1e-8)
    assert m.enc_dict["datetime"].weight.grad.dtype == torch.bfloat16


def test_calendar_rejects_accidental_blanket_bf16_cast():
    m = model().bfloat16()
    with pytest.raises(ValueError, match="set_compute_dtype"):
        m(batch(), return_embeddings=False)


@pytest.mark.parametrize("encoding", ["scalar", "calendar"])
def test_checkpoint_roundtrip_and_missing_tag_legacy(tmp_path, encoding):
    m = model(encoding)
    if encoding == "calendar":
        with torch.no_grad():
            m.dec_dict["datetime"].weight.fill_(0.00123456789)
            m.dec_dict["datetime"].bias.fill_(1.001)
    config = {"model": DIMS.copy()}
    if encoding == "calendar":
        config["model"]["datetime_encoding"] = encoding
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_file(m.state_dict(), str(tmp_path / "model.safetensors"))
    loaded = net.RelationalTransformer.from_pretrained(tmp_path).set_compute_dtype()
    assert loaded.datetime_encoding == encoding
    for name, value in m.state_dict().items():
        assert value.dtype == loaded.state_dict()[name].dtype
        assert torch.equal(value, loaded.state_dict()[name])
    a = m(batch(), return_embeddings=False)
    b = loaded(batch(), return_embeddings=False)
    assert torch.equal(a[0], b[0])
    for key in a[1]:
        assert torch.equal(a[1][key], b[1][key])
    other = "calendar" if encoding == "scalar" else "scalar"
    with pytest.raises(ValueError, match="explicit warm-start"):
        net.RelationalTransformer.from_pretrained(tmp_path, datetime_encoding=other)
    changed = {**config["model"], "datetime_encoding": other}
    (tmp_path / "config.json").write_text(json.dumps({"model": changed}))
    error = ValueError if encoding == "scalar" else RuntimeError
    reason = "stored FP32" if encoding == "scalar" else "size mismatch"
    with pytest.raises(error, match=reason):
        net.RelationalTransformer.from_pretrained(tmp_path)


def test_calendar_forward_captures_as_single_cpu_graph():
    m = model()
    b = batch()
    expected = m(b, return_embeddings=False)
    compiled = torch.compile(m, backend="eager", fullgraph=True)
    observed = compiled(b, return_embeddings=False)
    assert torch.equal(expected[0], observed[0])
    for key in expected[1]:
        assert torch.equal(expected[1][key], observed[1][key])


def test_numeric_loss_is_fp32_without_changing_numeric_head():
    m = model()
    b = batch()
    b["is_targets"][0, 0] = False
    b["number_values"][0, 2, 0] = 0.1
    loss, outputs, types, _ = m(b, return_embeddings=False)
    represented_target = b["number_values"][0, 2, 0]
    expected = F.huber_loss(torch.tensor(0.0), represented_target.float())
    rounded = F.huber_loss(torch.tensor(0.0).bfloat16(), represented_target)
    assert outputs["number"].dtype == torch.bfloat16
    assert torch.equal(loss, expected)
    assert torch.equal(types["number"], expected)
    assert loss.item() != rounded.item()


@pytest.mark.parametrize("loss_fn", ["huber", "l1", "bce"])
def test_fp32_head_matmul_reductions_and_gradients_match_double_reference(loss_fn):
    m = model(loss_fn=loss_fn)
    head = m.dec_dict["datetime"]
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[0.1001234, -0.2002345, 0.3003456, 0.4004567]]))
        head.bias.fill_(0.010123)
    x = (torch.arange(32).reshape(2, 4, 4).remainder(11) - 5).bfloat16() / 4
    x.requires_grad_()
    sem_types = torch.tensor([[2, 2, 0, 2], [2, 2, 1, 2]])
    targets = torch.ones((2, 4), dtype=torch.bool)
    padding = torch.tensor([[False, False, False, False], [False, False, False, True]])
    valid = torch.tensor([[True, True, False, False], [True, True, False, True]])
    labels = {
        "datetime": torch.tensor(
            [
                [[0.12345], [1.78901], [float("nan")], [float("nan")]],
                [[-0.23456], [-2.34567], [float("nan")], [float("nan")]],
            ]
        ),
        "number": torch.full((2, 4, 1), 0.12345, dtype=torch.bfloat16),
        "text": torch.full((2, 4, 2), -0.23456, dtype=torch.bfloat16),
    }
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, predictions, per_type, mask = m._calendar_loss(
            x, labels, sem_types, targets, padding, valid
        )
    w = head.weight.detach().double().requires_grad_()
    bias = head.bias.detach().double().requires_grad_()
    reference_date = F.linear(x.detach().double(), w, bias)
    torch.testing.assert_close(
        predictions["datetime"].double(), reference_date, atol=1e-7, rtol=1e-6
    )
    assert not torch.equal(
        predictions["datetime"], predictions["datetime"].bfloat16().float()
    )
    expected_mask = torch.tensor([[True, True, True, False], [True, True, True, False]])
    assert torch.equal(mask, expected_mask)
    cell_losses = torch.zeros((2, 4), dtype=torch.float64)
    for index, name in enumerate(net.SEM_TYPE_NAMES):
        chosen = (sem_types == index) & expected_mask
        prediction = (
            reference_date
            if name == "datetime"
            else predictions[name].detach().double()
        )
        prediction = prediction[chosen]
        target = labels[name].double()[chosen]
        if loss_fn == "l1":
            value = (prediction - target).abs().mean(-1)
        elif loss_fn == "huber" or name == "datetime":
            residual = (prediction - target).abs()
            value = torch.where(
                residual < 1, residual.square() / 2, residual - 0.5
            ).mean(-1)
        else:
            value = F.binary_cross_entropy_with_logits(
                prediction, (target > 0).double(), reduction="none"
            ).mean(-1)
        cell_losses[chosen] = value
        torch.testing.assert_close(
            per_type[name].double(), value.mean(), atol=2e-7, rtol=1e-6
        )
    expected_loss = (cell_losses.sum(1) / expected_mask.sum(1)).mean()
    torch.testing.assert_close(loss.double(), expected_loss, atol=2e-7, rtol=1e-6)
    expected_gradients = torch.autograd.grad(expected_loss, (w, bias))
    loss.backward()
    for parameter, reference in zip(
        (head.weight, head.bias), expected_gradients, strict=True
    ):
        assert parameter.grad.dtype == torch.float32
        torch.testing.assert_close(
            parameter.grad.double(), reference, atol=2e-7, rtol=1e-6
        )
    assert x.grad.dtype == torch.bfloat16
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("encoding", ["scalar", "calendar"])
def test_predict_rejects_datetime_targets_before_forward(monkeypatch, encoding):
    m = model(encoding)
    b = batch()
    b["is_targets"][0, 2] = False

    def forbidden_forward(*args, **kwargs):
        pytest.fail("datetime target reached the numeric prediction forward")

    monkeypatch.setattr(m, "forward", forbidden_forward)
    with pytest.raises(ValueError, match="datetime targets.*reconstruction loss"):
        m.predict(b, [4], "cpu", None)


@pytest.mark.parametrize("encoding", ["scalar", "calendar"])
def test_predict_numeric_with_datetime_context_keeps_previous_result(encoding):
    m = model(encoding)
    b = batch()
    b["is_targets"][0, 0] = False
    with torch.no_grad():
        m.dec_dict["number"].bias.fill_(3)
        m.dec_dict["datetime"].bias.fill_(7)
    observed = m.predict(b, [2, 4], "cpu", None)
    assert torch.equal(observed[2], torch.tensor([0.0], dtype=torch.bfloat16))
    assert torch.equal(observed[4], torch.tensor([3.0], dtype=torch.bfloat16))


def test_predict_ignores_padded_datetime_target_slots():
    m = model()
    b = batch()
    b["is_padding"][0, 0] = True
    b["datetime_valid"][0, 0] = False
    with torch.no_grad():
        m.dec_dict["number"].bias.fill_(3)
    assert m.predict(b, [4], "cpu", None)[4].item() == 3.0


@pytest.mark.parametrize("explicit_encoding", [False, True])
def test_standalone_calendar_checkpoint_uses_file_tag(tmp_path, explicit_encoding):
    m = model()
    with torch.no_grad():
        m.dec_dict["datetime"].weight.fill_(0.00123456789)
        m.dec_dict["datetime"].bias.fill_(1.001)
    path = tmp_path / "standalone.safetensors"
    checkpoints.save_model(m.state_dict(), path, {"datetime_encoding": "calendar"})
    assert not (tmp_path / "config.json").exists()
    kwargs = {"datetime_encoding": "calendar"} if explicit_encoding else {}
    loaded = net.RelationalTransformer.from_pretrained(
        path, **DIMS, **kwargs
    ).set_compute_dtype()
    assert loaded.datetime_encoding == "calendar"
    assert loaded.config["model"]["datetime_encoding"] == "calendar"
    for name, value in m.state_dict().items():
        assert loaded.state_dict()[name].dtype == value.dtype
        assert torch.equal(loaded.state_dict()[name], value)


@pytest.mark.parametrize("directory", [False, True])
def test_file_encoding_fills_missing_config_tag_and_respects_checkpoint_filename(
    tmp_path, directory
):
    path = tmp_path / "swa_steps=4.safetensors"
    (tmp_path / "config.json").write_text(
        json.dumps({"model": DIMS, "checkpoint_file": path.name})
    )
    checkpoints.save_model(
        model().state_dict(), path, {"datetime_encoding": "calendar"}
    )
    loaded = net.RelationalTransformer.from_pretrained(tmp_path if directory else path)
    assert loaded.datetime_encoding == "calendar"
    assert loaded.enc_dict["datetime"].in_features == 19


@pytest.mark.parametrize(
    "config_encoding, file_encoding", [("scalar", "calendar"), ("calendar", "scalar")]
)
def test_checkpoint_encoding_conflicts_fail_before_loading(
    tmp_path, config_encoding, file_encoding
):
    path = tmp_path / "model.safetensors"
    (tmp_path / "config.json").write_text(
        json.dumps({"model": {**DIMS, "datetime_encoding": config_encoding}})
    )
    checkpoints.save_model(
        model(file_encoding).state_dict(), path, {"datetime_encoding": file_encoding}
    )
    with pytest.raises(ValueError, match="conflicting datetime_encoding"):
        checkpoints.resolve_checkpoint(path)
    with pytest.raises(ValueError, match="conflicting datetime_encoding"):
        net.RelationalTransformer.from_pretrained(tmp_path)
    with pytest.raises(ValueError, match="conflicting datetime_encoding"):
        checkpoints.load_model(path, datetime_encoding=config_encoding)


@pytest.mark.parametrize("tag_source", ["config", "file"])
def test_unknown_checkpoint_encoding_is_rejected(tmp_path, tag_source):
    path = tmp_path / "model.safetensors"
    config = {"model": DIMS.copy()}
    metadata = {}
    if tag_source == "config":
        config["model"]["datetime_encoding"] = "unknown"
    else:
        metadata["datetime_encoding"] = "unknown"
    (tmp_path / "config.json").write_text(json.dumps(config))
    checkpoints.save_model(model().state_dict(), path, metadata)
    with pytest.raises(ValueError, match="unknown datetime_encoding"):
        net.RelationalTransformer.from_pretrained(path)


def test_untagged_standalone_checkpoint_remains_scalar_and_accepts_bf16(tmp_path):
    m = model("scalar")
    path = tmp_path / "legacy.safetensors"
    checkpoints.save_model(m.state_dict(), path)
    config, resolved = checkpoints.resolve_checkpoint(path)
    assert config == {} and resolved == path
    loaded = net.RelationalTransformer.from_pretrained(path, **DIMS).set_compute_dtype()
    assert loaded.datetime_encoding == "scalar"
    for name, value in m.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)


def test_untagged_calendar_shape_never_infers_encoding(tmp_path):
    path = tmp_path / "unmarked.safetensors"
    checkpoints.save_model(model().state_dict(), path)
    with pytest.raises(ValueError, match="explicit warm-start"):
        net.RelationalTransformer.from_pretrained(
            path, **DIMS, datetime_encoding="calendar"
        )
    with pytest.raises(RuntimeError, match="size mismatch"):
        net.RelationalTransformer.from_pretrained(path, **DIMS)


@pytest.mark.parametrize("name", ["dec_dict.datetime.weight", "dec_dict.datetime.bias"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tag_source", ["config", "file"])
def test_calendar_load_and_warmstart_reject_narrowed_source_heads(
    tmp_path, name, dtype, tag_source
):
    m = model()
    with torch.no_grad():
        m.dec_dict["datetime"].weight.fill_(0.00123456789)
        m.dec_dict["datetime"].bias.fill_(1.001)
    state = m.state_dict()
    state[name] = state[name].to(dtype)
    path = tmp_path / "model.safetensors"
    config = {"model": DIMS.copy()}
    metadata = {}
    if tag_source == "config":
        config["model"]["datetime_encoding"] = "calendar"
    else:
        metadata["datetime_encoding"] = "calendar"
    (tmp_path / "config.json").write_text(json.dumps(config))
    checkpoints.save_model(state, path, metadata)
    # Shared validator applies before load_state_dict can silently widen sources.
    with pytest.raises(ValueError, match="stored FP32 " + name):
        checkpoints.validate_datetime_state_dict(state, "calendar")
    # Trainer passes the resolved config mode, including for config-only tags.
    config, resolved = checkpoints.resolve_checkpoint(path)
    with pytest.raises(ValueError, match="stored FP32 " + name):
        checkpoints.load_model(
            resolved, datetime_encoding=config["model"]["datetime_encoding"]
        )
    with pytest.raises(ValueError, match="stored FP32 " + name):
        net.RelationalTransformer.from_pretrained(path)
    if tag_source == "file":
        with pytest.raises(ValueError, match="stored FP32 " + name):
            checkpoints.load_model(path)
    assert state[name].dtype == dtype  # Validation never rewrites source tensors.


def test_shared_head_validator_keeps_scalar_compatibility_and_requires_calendar_keys():
    state = model("scalar").state_dict()
    checkpoints.validate_datetime_state_dict(state, "scalar")
    state = model().state_dict()
    del state["dec_dict.datetime.bias"]
    with pytest.raises(ValueError, match="missing dec_dict.datetime.bias"):
        checkpoints.validate_datetime_state_dict(state, "calendar")
    with pytest.raises(ValueError, match="unknown datetime_encoding"):
        checkpoints.validate_datetime_state_dict(state, "other")


@pytest.mark.parametrize("tag_source", ["config", "file"])
def test_old14_calendar_checkpoints_rejected_before_load_or_warmstart(
    tmp_path, tag_source
):
    state = model().state_dict()
    state["enc_dict.datetime.weight"] = state["enc_dict.datetime.weight"][
        :, :14
    ].clone()
    path = tmp_path / "model.safetensors"
    config = {"model": DIMS.copy()}
    metadata = {}
    if tag_source == "config":
        config["model"]["datetime_encoding"] = "calendar"
    else:
        metadata["datetime_encoding"] = "calendar"
    (tmp_path / "config.json").write_text(json.dumps(config))
    checkpoints.save_model(state, path, metadata)
    with pytest.raises(ValueError, match="obsolete 14-coordinate.*requires 19"):
        checkpoints.validate_datetime_state_dict(state, "calendar")
    resolved_config, resolved_path = checkpoints.resolve_checkpoint(path)
    with pytest.raises(ValueError, match="obsolete 14-coordinate.*requires 19"):
        checkpoints.load_model(
            resolved_path,
            datetime_encoding=resolved_config["model"]["datetime_encoding"],
        )
    with pytest.raises(ValueError, match="obsolete 14-coordinate.*requires 19"):
        net.RelationalTransformer.from_pretrained(path)
    # Raw PyTorch loading is still strict even without the public file loader.
    with pytest.raises(
        RuntimeError, match="size mismatch for enc_dict.datetime.weight"
    ):
        model().load_state_dict(state)
    assert state["enc_dict.datetime.weight"].shape[1] == 14


@pytest.mark.parametrize("precision", ["ieee", "none", "tf32", "unknown", None])
def test_cuda_matmul_precision_uses_effective_modern_policy(monkeypatch, precision):
    class ModernMatmul:
        fp32_precision = precision

        @property
        def allow_tf32(self):
            pytest.fail("modern policy must not consult legacy getters")

    def legacy_getter():
        pytest.fail("modern policy must not consult legacy getters")

    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", raising=False)
    monkeypatch.setattr(torch.backends.cuda, "matmul", ModernMatmul())
    monkeypatch.setattr(torch, "get_float32_matmul_precision", legacy_getter)
    if precision in ("ieee", "none"):
        net.validate_cuda_matmul_precision()
    else:
        with pytest.raises(ValueError, match="effective .*fp32_precision"):
            net.validate_cuda_matmul_precision()


@pytest.mark.parametrize(
    "allow_tf32, precision, accepted",
    [
        (False, "highest", True),
        (True, "highest", False),
        (False, "high", False),
        (False, "medium", False),
    ],
)
def test_cuda_matmul_precision_legacy_fallback(
    monkeypatch, allow_tf32, precision, accepted
):
    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", raising=False)
    monkeypatch.setattr(
        torch.backends.cuda, "matmul", SimpleNamespace(allow_tf32=allow_tf32)
    )
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: precision)
    if accepted:
        net.validate_cuda_matmul_precision()
    else:
        with pytest.raises(ValueError, match="'highest' precision with TF32 disabled"):
            net.validate_cuda_matmul_precision()


def test_cuda_matmul_precision_does_not_hide_modern_getter_errors(monkeypatch):
    class BrokenMatmul:
        @property
        def fp32_precision(self):
            raise RuntimeError("broken backend policy")

        @property
        def allow_tf32(self):
            pytest.fail("a modern getter error must not fall back to legacy policy")

    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", raising=False)
    monkeypatch.setattr(torch.backends.cuda, "matmul", BrokenMatmul())
    with pytest.raises(RuntimeError, match="broken backend policy"):
        net.validate_cuda_matmul_precision()


def test_cuda_matmul_precision_rejects_forced_override_before_getters(monkeypatch):
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    monkeypatch.setattr(torch.backends.cuda, "matmul", object())
    with pytest.raises(ValueError, match="OVERRIDE=1 forces TF32"):
        net.validate_cuda_matmul_precision()
    assert os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] == "1"


@pytest.mark.parametrize("encoding", ["scalar", "calendar"])
def test_cpu_dtype_setup_does_not_require_cuda_policy(monkeypatch, encoding):
    def reject():
        pytest.fail("CPU setup must not inspect CUDA precision")

    monkeypatch.setattr(net, "validate_cuda_matmul_precision", reject)
    m = model(encoding)
    assert m.dec_dict["datetime"].weight.device.type == "cpu"


@pytest.mark.parametrize("accepted", [True, False])
def test_calendar_cuda_setup_validates_before_any_cast(monkeypatch, accepted):
    # Device metadata only: this checks dispatch/order without CUDA allocation.
    events = []

    def validate():
        events.append("validate")
        if not accepted:
            raise ValueError("unsafe policy")

    body = SimpleNamespace(to=lambda **kwargs: events.append("cast"))
    m = SimpleNamespace(
        datetime_encoding="calendar",
        dec_dict={
            "datetime": SimpleNamespace(
                weight=SimpleNamespace(device=torch.device("cuda"))
            )
        },
        named_children=lambda: [("body", body)],
    )
    monkeypatch.setattr(net, "validate_cuda_matmul_precision", validate)
    if accepted:
        assert net.RelationalTransformer.set_compute_dtype(m) is m
        assert events == ["validate", "cast"]
    else:
        with pytest.raises(ValueError, match="unsafe policy"):
            net.RelationalTransformer.set_compute_dtype(m)
        assert events == ["validate"]


@pytest.mark.parametrize(
    "setup, rejected, needs_modern",
    [
        pytest.param("pass", False, False, id="default"),
        pytest.param(
            "torch.set_float32_matmul_precision('highest')",
            False,
            False,
            id="legacy-highest",
        ),
        pytest.param(
            "torch.set_float32_matmul_precision('high')",
            True,
            False,
            id="legacy-high",
        ),
        pytest.param(
            "torch.set_float32_matmul_precision('medium')",
            True,
            False,
            id="legacy-medium",
        ),
        pytest.param(
            "torch.backends.cuda.matmul.allow_tf32 = True",
            True,
            False,
            id="legacy-tf32",
        ),
        pytest.param(
            "torch.backends.cuda.matmul.fp32_precision = 'tf32'",
            True,
            True,
            id="modern-matmul-tf32",
        ),
        pytest.param(
            "torch.backends.fp32_precision = 'tf32'",
            True,
            True,
            id="modern-inherited-tf32",
        ),
        pytest.param(
            "torch.backends.fp32_precision = 'tf32'\n"
            "torch.backends.cuda.matmul.fp32_precision = 'ieee'",
            False,
            True,
            id="modern-matmul-ieee-overrides-parent",
        ),
        pytest.param(
            "torch.backends.cudnn.conv.fp32_precision = 'tf32'",
            False,
            True,
            id="modern-unrelated-backend",
        ),
        pytest.param(
            "os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '1'",
            True,
            False,
            id="forced-env-override",
        ),
    ],
)
def test_cuda_matmul_precision_real_getters_without_cuda_init(
    setup, rejected, needs_modern
):
    # Each case starts fresh and uses only one setter family. The guard itself
    # only reads policy. In particular, do not read legacy getters when modern
    # per-backend overrides make those getters inconsistent or unsupported.
    script = f"""
import json
import os
import torch
from rt.model.net import validate_cuda_matmul_precision
assert not torch.cuda.is_initialized()
modern = hasattr(torch.backends.cuda.matmul, "fp32_precision")
if {needs_modern!r} and not modern:
    raise SystemExit(77)
{setup}
def snapshot():
    if modern:
        return torch.backends.cuda.matmul.fp32_precision
    return (torch.backends.cuda.matmul.allow_tf32,
            torch.get_float32_matmul_precision())
before = snapshot()
rejected = False
try:
    validate_cuda_matmul_precision()
except ValueError as exc:
    assert "IEEE CUDA matmul" in str(exc), str(exc)
    rejected = True
assert rejected is {rejected!r}, (before, rejected)
assert snapshot() == before
assert not torch.cuda.is_initialized()
print(json.dumps({{"policy": before, "rejected": rejected}}))
"""
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    env.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    if result.returncode == 77:
        pytest.skip("Torch has no backend-specific FP32 precision API")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["rejected"] is rejected
