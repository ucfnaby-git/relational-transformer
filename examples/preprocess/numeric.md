# Numeric features (experimental)

`numeric_encoding="bits"` exposes the existing normalized archive FP32 values
through bounded BF16 context features and an FP32 numeric head/target/loss path.
It requires no numeric sidecar or archive rewrite. This is an experimental
representation, not a demonstrated accuracy improvement or a recommendation
for full pretraining.

## Enabling the representation

| Setting | Meaning |
|---|---|
| `numeric_encoding="scalar"` | Default model behavior and legacy numeric head/labels |
| `numeric_encoding="bits"` | Bounded5 context and FP32 numeric head/labels |
| `exact_numeric=True` | Dataset/native sampler emits the additional fields |

Training and evaluation entry points enable `exact_numeric` for bits models.
Custom dataset/sampler callers must enable it explicitly. Its default is false;
legacy fields and sampling behavior remain unchanged when it is disabled.
A scalar model may receive exact numeric fields and ignores them.

Numeric and datetime settings are independent. Both numeric encodings work
with `datetime_encoding="scalar"` or `"calendar"`. Calendar mode still requires
its own `exact_datetime` data and [datetime sidecars](datetime.md).

## Stored values and model inputs

For a finite archived FP32 value `x`, let `u` be its unsigned 32-bit pattern.
The five BF16 coordinates, in order, are:

```text
BF16(F64(x) / (1 + abs(F64(x)))),
((u >> 24) & 255) / 256, ((u >> 16) & 255) / 256,
((u >>  8) & 255) / 256, ( u        & 255) / 256
```

The first coordinate is explicitly +0 for FP32 subnormals and signed zeros,
detected from exponent bits before arithmetic. It is an approximate monotone
cue, bounded by one in magnitude. The four byte digits are exactly representable
normal BF16 values or zero; they reconstruct all finite source FP32 bit patterns,
including signed zero and subnormals. Byte carries introduce discontinuities.

| Batch field | Shape and dtype |
|---|---|
| `number_features` | `[B, S, 5]`, BF16 |
| `number_target_values` | `[B, S, 1]`, FP32 from the archive before BF16 narrowing |

The existing BF16 `number_values` field remains available. Boolean cells mapped
to Number use their stored FP32 boolean values; `legacy_boolean=True` retains
the previous Boolean semantics and leaves new numeric slots zero.

NaN omission and missing-target handling follow the legacy sampler policy.
With `exact_numeric=True`, a selected infinite primary target or
selected infinite context value raises a fatal `ValueError` before emission,
including under `ignore_data_errors`. The error identifies the dataset, node, column and bits.
This is validation on consumption, not an archive-wide scan or certification
of unsampled cells.

New numeric fields are zero for padding and irrelevant cells. The model gathers
every field with the same cell sort and masks target/padding/irrelevant context before
projection. Retained target labels feed loss only. Existing semantic and padding
masks determine valid number cells; there is no additional numeric validity mask.

## Compute and checkpoints

Context encoders and the Transformer remain BF16. Bits mode retains FP32 numeric
head weights and executes that head with autocast disabled. Loss operands and
reductions use FP32 while preserving configured numeric Huber/L1/BCE semantics.
Calendar datetime independently retains its FP32 regression head/loss policy.

Capture authoritative FP32 master weights before `model.set_compute_dtype()`.
Use that method instead of a blanket BF16 cast. On CUDA it validates effective
IEEE matmul policy without changing global settings and rejects forced TF32.
Custom callers moving a model to CUDA afterwards must call
`rt.model.net.validate_cuda_matmul_precision()` before first compiled forward.
Keep the validated policy fixed through forward and backward.
Custom compiled AMP loops also need the compiler's backward-autocast policy
described in the [pretraining guide](../pretrain/README.md).

Serialize `numeric_encoding` independently of `datetime_encoding` in model
config and checkpoint metadata. Missing numeric tags mean scalar; unknown or
conflicting tags are rejected. `bits` identifies this fixed bounded5 v1 map;
a changed map must not silently reuse that contract. Loads are strict and
require stored FP32 weights/biases for each active precision head. Changing
encoding requires explicit checkpoint migration and fresh optimizer/SWA state.
Widening an already narrowed head cannot restore its original precision.

## Evaluation and precision limits

For scalar-versus-bits comparisons, set **`exact_numeric=True` on both evaluation
datasets**. The evaluator uses `number_target_values` whenever present, regardless
of model encoding, and preserves FP32 through selection, gathering and metrics.
When absent, the legacy label path remains unchanged. Use the same rows, splits
and archived normalized FP32 truth for both arms; changing label quantization
between arms is not an accuracy gain. Official RelBench export scoring uses
benchmark ground truth and is a separately identified metric.

The exact claim concerns archive FP32 bits before learned projection. It does
not restore precision lost in normalization, integer/FP64 conversion, or earlier
preprocessing. It does not guarantee invertible learned embeddings, exact scalar
predictions, or improved task accuracy.

Bounded inputs do not guarantee finite differences/reductions for arbitrary
FP32 targets. Opposite-sign extrema can overflow; tiny losses and subnormal
gradients can underflow or flush. Targets are not clipped and the head/loss is
not converted wholesale to FP64. Validate the actual normalized corpus domain
and off-BF16-grid supervision before small paired experiments.
