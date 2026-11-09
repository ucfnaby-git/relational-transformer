# Datetime sidecars

Calendar encoding reads the source timestamp before floating-point normalization.
It requires the raw database used to create the archive. Casting a released
FP32 scalar to a wider dtype cannot recover discarded timestamp information.

## Fresh preprocessing

```python
from rt.preprocess import one

one(
    dataset="data/my-collection/my-db",
    out_dir="data/my-collection-calendar",
    embedder="all-MiniLM-L12-v2",
    batch_size=1024,
    skip_tasks=False,
    embed=True,
    upload_repo=None,
    public=False,
    revision=None,
    exact_datetime=True,
)
```

Use a new output directory and keep raw inputs immutable during preprocessing.
The native operation records input hashes before and after the existing graph
preprocessor, then builds the sidecar. The default `exact_datetime=False` keeps
the existing preprocessing and sampling behavior.

## Stored values and model inputs

The original graph, text dictionary, and embedding matrix remain unchanged.
Three additional files are written:

| File | Content |
|---|---|
| `datetime_source.json` | Raw/archive identity receipt and preprocessing policy |
| `datetime_exact.bin` | Node offsets, signed int64 Unix seconds, uint32 nanosecond remainders |
| `datetime_exact.json` | Format, normalization, content hashes, and local file state |

The binary uses little-endian integers. Its `RTDTIME1` header describes the
arrays; metadata format version 2 binds it to a source receipt. Timestamps use
floor division for negative epochs, with remainders in `[0, 1_000_000_000)`.
Null timestamps remain absent cells. Coarse-unit timestamps are not expanded
into int64 nanoseconds, so dates outside that narrower range remain supported.

The native sampler derives UTC calendar components:

```
year // 65536, (year % 65536) // 256, year % 256,
month, day, hour, minute, second,
millisecond // 256, millisecond % 256,
microsecond // 256, microsecond % 256,
nanosecond // 256, nanosecond % 256
```

The subsecond groups are successive three-digit groups of the nanosecond
remainder. Every component is divided by 256, a power-of-two scaling exactly
representable in BF16 over Chrono's supported calendar range. The three year
components also cover negative and large years. Unsupported calendar ranges
are rejected instead of wrapped. Timestamp instants are interpreted in UTC;
this representation does not encode a separate local-time-zone identity.

Calendar batches retain integer source pairs for inspection and include
`datetime_features` (BF16, 14 channels), `datetime_valid` (bool), and
`datetime_target_values` (FP32, one channel). The legacy BF16 scalar field is
retained for compatibility and is not used by the calendar encoder.
Retained datetime cells whose legacy normalization produced NaN, such as
constant dates, are recovered from their source values. This can change sampled
context occupancy and random-mask draws relative to scalar mode; columns already
discarded by graph preprocessing are not recovered.

The reconstruction target is normalized separately from context features.
Population mean and standard deviation are computed in FP64 after subtracting
an integer reference timestamp. Training task tables are included;
validation/test task tables are excluded. For database tables with a declared
row-time column and a manifest validation cutoff, only rows at or before that
cutoff with a non-null row time contribute. Without a cutoff or a row-time
column, all retained datetime cells in the database table are eligible.
A constant or empty fit uses a standard
deviation of one second. The fitted statistics and scope are saved in metadata.

Raw pairs and calendar features preserve source information before projection.
The FP32 scalar target, learned projections, and predictions remain approximate.
This format does not promise invertible learned embeddings or nanosecond output
resolution. Existing row-time filtering and columns discarded by the graph
preprocessor are unchanged.
Datetime cells can be masked reconstruction targets. Public task prediction and
benchmark evaluation still require numeric targets and reject datetime targets;
they do not interpret the datetime head as a task prediction API.

## Historical archives

`backfill_datetime(..., source_receipt=path)` accepts an explicit, independently
trusted statement that a particular raw revision produced a particular archive.
Both revisions' content must match that statement. Matching row counts, keys,
or a replay of lossy preprocessing does not establish the original timestamps.

The receipt has these required fields; unknown fields are rejected:

| Field | Meaning |
|---|---|
| `format_version` | Integer `1` |
| `dataset_name` | Raw manifest/database name |
| `kind` | `historical` or `fresh_preprocess` |
| `preprocess` | `implementation`, `skip_tasks`, `skip_db`, `identifier_policy`, `raw_manifest_policy` |
| `origin` | Nonempty `issuer`, `raw_revision`, `archive_revision`, `evidence` |
| `raw_files` | Relative path to `{ "bytes": integer, "sha256": hex_digest }` |
| `base_archive` | Same identity map for the six files listed below |

`identifier_policy` is `none`, `empty`, `empty_or_time`, or `threshold`;
`raw_manifest_policy` is `all-layout-inputs-v1`. The input inventory includes
`manifest.yaml`, every `db/*.parquet`, and each task's `manifest.yaml` and
`train.parquet`, `val.parquet`, and `test.parquet` files that exist. Include
ignored tasks and tables too. Adding or removing an inventoried file invalidates
the receipt. Unrelated embeddings and auxiliary files are outside this policy.

The archive inventory must contain exactly `nodes.rkyv`, `offsets.rkyv`,
`p2f_adj.rkyv`, `column_index.json`, `table_info.json`, and `text.json`.
SHA-256 digests use 64 lowercase hexadecimal digits. Revisions must identify
immutable content; `main` is not an immutable revision.

Native validation checks these commitments and positional node/cell structure.
It cannot authenticate an issuer or prove a historical association from hashes
alone. Supply historical receipts only after independently reviewing their
evidence. Without that evidence, use fresh preprocessing. An omitted receipt
path reuses only a receipt from a trusted fresh preprocessing operation; its
`kind` string is not an authenticity mechanism. Explicit historical receipt
bytes are retained unchanged.

Fresh receipt revisions are `sha256:` followed by the digest of their respective
identity maps, serialized as compact UTF-8 JSON with lexicographically sorted
paths, fields ordered `bytes` then `sha256`, and no trailing newline. There is
no Unicode normalization. The implementation identity is
`pre.rs+common.rs:sha256:` followed by the hash of those embedded source files
concatenated in that order. This identifies preprocessing source, not a complete
hermetic compiler/dependency environment.

The current builder also requires a declared database primary key to equal its
zero-based physical row index. It rejects other layouts instead of guessing a
permutation. Task foreign keys, cell identities, row counts, and null masks are
checked against the archive.

## Copying and validation

After copying a database, run `rt.preprocess.validate_datetime(pre_dir=...)`
before starting workers. It verifies hashes and complete node/cell structure,
requires no raw directory, and refreshes local modification timestamps only
after successful validation. The metadata directory must be writable. A content
or structural failure leaves the published files unchanged. Normal workers use cached file-state checks rather
than rehashing large archives per worker; same-size, same-mtime external
tampering is outside that cache contract.

Keep archives and sidecars immutable while workers have them mapped. Exact
writers use a lock and publish metadata last. Each rename is atomic, but the
three-file publication is not a transaction: an interrupted replacement can
leave mixed files that validation rejects. Rebuild from admitted source data
in a new output directory if recovery is needed. Do not repair a failed
validation by replacing expected hashes with hashes of the changed files.
