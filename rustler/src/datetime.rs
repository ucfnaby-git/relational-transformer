//! Exact raw timestamps alongside the unchanged version-1 rkyv archive.
//!
//! Backfill requires an observed fresh-build receipt or explicitly trusted
//! historical provenance. Content hashes do not authenticate historical claims.
//! No inverse z-score, row permutation inference, or legacy replay is used.
use crate::common::{ArchivedNode, ArchivedOffsets, ArchivedSemType, TableInfo};
use chrono::{DateTime, Datelike, NaiveDate, Timelike};
use half::bf16;
use memmap2::Mmap;
use polars::prelude::*;
use rkyv::rancor::Error;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const FEATURES: usize = 14;
const ENCODING: &str = "calendar14-v1";
const META: &str = "datetime_exact.json";
const DATA: &str = "datetime_exact.bin";
const SOURCE: &str = "datetime_source.json";
const RAW_POLICY: &str = "all-layout-inputs-v1";
const MAGIC: &[u8; 8] = b"RTDTIME1";
const HEADER: usize = 64;
const BASE_FILES: &[&str] = &[
    "nodes.rkyv",
    "offsets.rkyv",
    "p2f_adj.rkyv",
    "column_index.json",
    "table_info.json",
    "text.json",
];
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn require(ok: bool, message: impl Into<String>) -> Result<()> {
    if !ok {
        return Err(std::io::Error::other(message.into()).into());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Timestamp {
    pub seconds: i64,
    pub nanoseconds: u32,
}

impl Timestamp {
    pub fn from_ticks(value: i64, unit: TimeUnit) -> Self {
        let q = match unit {
            TimeUnit::Milliseconds => 1_000,
            TimeUnit::Microseconds => 1_000_000,
            TimeUnit::Nanoseconds => 1_000_000_000,
        };
        Self {
            seconds: value.div_euclid(q),
            nanoseconds: (value.rem_euclid(q) * (1_000_000_000 / q)) as u32,
        }
    }

    fn from_value(value: AnyValue<'_>) -> Result<Option<Self>> {
        match value {
            AnyValue::Null => Ok(None),
            AnyValue::Datetime(v, unit, _) => Ok(Some(Self::from_ticks(v, unit))),
            AnyValue::Date(days) => Ok(Some(Self {
                seconds: i64::from(days) * 86_400,
                nanoseconds: 0,
            })),
            value => Err(std::io::Error::other(format!(
                "expected a temporal value, got {value:?}"
            ))
            .into()),
        }
    }

    pub fn features(self) -> Result<[bf16; FEATURES]> {
        require(
            self.nanoseconds < 1_000_000_000,
            "nanoseconds must be < 1e9",
        )?;
        let dt = DateTime::from_timestamp(self.seconds, self.nanoseconds).ok_or_else(|| {
            std::io::Error::other(format!(
                "timestamp ({}, {}) is outside Chrono's supported calendar range",
                self.seconds, self.nanoseconds
            ))
        })?;
        let year = dt.year();
        let ns = self.nanoseconds;
        let ms_part = ns / 1_000_000;
        let us_part = ns / 1_000 % 1_000;
        let ns_part = ns % 1_000;
        let values = [
            year.div_euclid(65_536),
            year.rem_euclid(65_536) / 256,
            year.rem_euclid(256),
            dt.month() as i32,
            dt.day() as i32,
            dt.hour() as i32,
            dt.minute() as i32,
            dt.second() as i32,
            (ms_part / 256) as i32,
            (ms_part % 256) as i32,
            (us_part / 256) as i32,
            (us_part % 256) as i32,
            (ns_part / 256) as i32,
            (ns_part % 256) as i32,
        ];
        Ok(values.map(|v| bf16::from_f32(v as f32 / 256.0)))
    }

    fn offset_from(self, origin: Self) -> f64 {
        // Subtract integers first, including at the signed i64 endpoints.
        (i128::from(self.seconds) - i128::from(origin.seconds)) as f64
            + (i64::from(self.nanoseconds) - i64::from(origin.nanoseconds)) as f64 * 1e-9
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Normalization {
    pub encoding: String,
    pub scope: String,
    pub count: u64,
    pub origin_seconds: i64,
    pub origin_nanoseconds: u32,
    pub mean_offset_seconds: f64,
    pub std_seconds: f64,
}

impl Normalization {
    pub fn target(&self, timestamp: Timestamp) -> f32 {
        let origin = Timestamp {
            seconds: self.origin_seconds,
            nanoseconds: self.origin_nanoseconds,
        };
        ((timestamp.offset_from(origin) - self.mean_offset_seconds) / self.std_seconds) as f32
    }

    fn validate(&self) -> Result<()> {
        require(
            self.encoding == "raw_training_seconds_v1"
                && self.origin_nanoseconds < 1_000_000_000
                && self.mean_offset_seconds.is_finite()
                && self.std_seconds.is_finite()
                && self.std_seconds > 0.0,
            "invalid exact datetime normalization",
        )
    }
}

#[derive(Default)]
struct Moments {
    origin: Option<Timestamp>,
    count: u64,
    mean: f64,
    m2: f64,
}

impl Moments {
    fn push(&mut self, value: Timestamp) {
        let origin = *self.origin.get_or_insert(value);
        let value = value.offset_from(origin);
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (value - self.mean);
    }

    fn finish(self) -> Normalization {
        let origin = self.origin.unwrap_or(Timestamp {
            seconds: 0,
            nanoseconds: 0,
        });
        let scale = (self.m2.max(0.0) / self.count.max(1) as f64).sqrt();
        Normalization {
            encoding: "raw_training_seconds_v1".into(),
            scope: "retained DateTime cells: Db at/before val_timestamp plus Train; no Val/Test"
                .into(),
            count: self.count,
            origin_seconds: origin.seconds,
            origin_nanoseconds: origin.nanoseconds,
            mean_offset_seconds: self.mean,
            std_seconds: if scale.is_finite() && scale > 0.0 {
                scale
            } else {
                1.0
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContentIdentity {
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    bytes: u64,
    modified_ns: u128,
    sha256: String,
}
impl Identity {
    fn content(&self) -> ContentIdentity {
        ContentIdentity {
            bytes: self.bytes,
            sha256: self.sha256.clone(),
        }
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_state(path: &Path) -> Result<(u64, u128)> {
    let metadata = path.metadata()?;
    Ok((
        metadata.len(),
        metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos(),
    ))
}

fn identity(path: &Path) -> Result<Identity> {
    let before = file_state(path)?;
    let mut reader = BufReader::new(File::open(path)?);
    let mut hash = Sha256::new();
    let mut buf = vec![0; 1024 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    require(
        before == file_state(path)?,
        format!("file changed while reading {}", path.display()),
    )?;
    Ok(Identity {
        bytes: before.0,
        modified_ns: before.1,
        sha256: format!("{:x}", hash.finalize()),
    })
}

fn check_identity(path: &Path, expected: &Identity) -> Result<()> {
    require(
        file_state(path)? == (expected.bytes, expected.modified_ns),
        format!(
            "stale datetime sidecar: {} file state changed; run validate_datetime(pre_dir)",
            path.display()
        ),
    )?;
    // Never hash a multi-GB artifact in each worker. Explicit validation performs
    // the full content scan after copying/downloading and refreshes local times.
    if expected.bytes <= 1_048_576 {
        require(
            identity(path)?.sha256 == expected.sha256,
            format!("stale datetime sidecar: {} content changed", path.display()),
        )?;
    }
    Ok(())
}

fn content_map(files: &BTreeMap<String, Identity>) -> BTreeMap<String, ContentIdentity> {
    files
        .iter()
        .map(|(name, id)| (name.clone(), id.content()))
        .collect()
}

fn verify_files(
    root: &Path,
    expected: &BTreeMap<String, ContentIdentity>,
) -> Result<BTreeMap<String, Identity>> {
    expected
        .iter()
        .map(|(name, expected)| {
            let path = root.join(name);
            let actual = identity(&path)?;
            require(
                actual.content() == *expected,
                format!("source/archive content mismatch: {}", path.display()),
            )?;
            Ok((name.clone(), actual))
        })
        .collect()
}

fn base_identities(pre: &Path) -> Result<BTreeMap<String, Identity>> {
    BASE_FILES
        .iter()
        .map(|name| Ok(((*name).into(), identity(&pre.join(name))?)))
        .collect()
}

fn stable_files(root: &Path, files: &BTreeMap<String, Identity>) -> Result<()> {
    for (name, id) in files {
        require(
            file_state(&root.join(name))? == (id.bytes, id.modified_ns),
            format!(
                "file changed during datetime operation: {}",
                root.join(name).display()
            ),
        )?;
    }
    Ok(())
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PreprocessConfig {
    implementation: String,
    skip_tasks: bool,
    skip_db: bool,
    identifier_policy: String,
    raw_manifest_policy: String,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Origin {
    issuer: String,
    raw_revision: String,
    archive_revision: String,
    evidence: String,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceReceipt {
    format_version: u32,
    dataset_name: String,
    kind: String,
    preprocess: PreprocessConfig,
    origin: Origin,
    raw_files: BTreeMap<String, ContentIdentity>,
    base_archive: BTreeMap<String, ContentIdentity>,
}

fn normal_name(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    matches!(parts.next(), Some(std::path::Component::Normal(_))) && parts.next().is_none()
}

fn raw_path_allowed(name: &str) -> bool {
    let parts: Vec<_> = name.split('/').collect();
    parts.iter().all(|s| normal_name(s))
        && match parts.as_slice() {
            ["manifest.yaml"] => true,
            ["db", file] => file.ends_with(".parquet"),
            [
                "tasks",
                _,
                "manifest.yaml" | "train.parquet" | "val.parquet" | "test.parquet",
            ] => true,
            _ => false,
        }
}

fn validate_content_ids(files: &BTreeMap<String, ContentIdentity>) -> Result<()> {
    for id in files.values() {
        require(
            id.sha256.len() == 64
                && id
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "invalid SHA-256 content identity",
        )?;
    }
    Ok(())
}

impl SourceReceipt {
    fn validate(&self) -> Result<()> {
        require(
            self.format_version == 1 && normal_name(&self.dataset_name),
            "invalid source receipt version/dataset",
        )?;
        require(
            self.kind == "fresh_preprocess" || self.kind == "historical",
            "unsupported source receipt kind",
        )?;
        require(
            self.preprocess.raw_manifest_policy == RAW_POLICY
                && !self.preprocess.implementation.trim().is_empty()
                && matches!(
                    self.preprocess.identifier_policy.as_str(),
                    "none" | "empty" | "empty_or_time" | "threshold"
                ),
            "unsupported source receipt preprocessing policy",
        )?;
        require(
            [
                &self.origin.issuer,
                &self.origin.raw_revision,
                &self.origin.archive_revision,
                &self.origin.evidence,
            ]
            .iter()
            .all(|s| !s.trim().is_empty()),
            "source receipt requires issuer, immutable revision identities and independent evidence",
        )?;
        require(
            self.raw_files.contains_key("manifest.yaml")
                && self.raw_files.keys().all(|p| raw_path_allowed(p)),
            "invalid source receipt raw manifest",
        )?;
        require(
            self.base_archive
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                == BASE_FILES.iter().copied().collect(),
            "incomplete source receipt archive identity",
        )?;
        validate_content_ids(&self.raw_files)?;
        validate_content_ids(&self.base_archive)?;
        if self.kind == "fresh_preprocess" {
            require(
                self.origin.issuer == "rt.rustler.preprocess"
                    && self.origin.raw_revision
                        == format!("sha256:{}", digest(&serde_json::to_vec(&self.raw_files)?))
                    && self.origin.archive_revision
                        == format!(
                            "sha256:{}",
                            digest(&serde_json::to_vec(&self.base_archive)?)
                        ),
                "invalid observed fresh-build receipt binding",
            )?;
        }
        Ok(())
    }
}

// Bounded inventory matching the documented raw layout. Include ignored files
// too, so skip flags never silently weaken the source revision commitment.
fn raw_paths(raw: &Path) -> Result<BTreeSet<String>> {
    let mut paths = BTreeSet::from(["manifest.yaml".to_owned()]);
    let db = raw.join("db");
    if db.is_dir() {
        for entry in fs::read_dir(db)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|s| s == "parquet") {
                let name = entry
                    .file_name()
                    .into_string()
                    .map_err(|_| std::io::Error::other("non-UTF8 raw filename"))?;
                paths.insert(format!("db/{name}"));
            }
        }
    }
    let tasks = raw.join("tasks");
    if tasks.is_dir() {
        for entry in fs::read_dir(tasks)? {
            let entry = entry?;
            if !entry.path().is_dir() {
                continue;
            }
            let task = entry
                .file_name()
                .into_string()
                .map_err(|_| std::io::Error::other("non-UTF8 task name"))?;
            for file in [
                "manifest.yaml",
                "train.parquet",
                "val.parquet",
                "test.parquet",
            ] {
                if entry.path().join(file).is_file() {
                    paths.insert(format!("tasks/{task}/{file}"));
                }
            }
        }
    }
    Ok(paths)
}

fn stable_raw(raw: &Path, files: &BTreeMap<String, Identity>) -> Result<()> {
    require(
        raw_paths(raw)? == files.keys().cloned().collect(),
        "raw source file manifest changed or is incomplete",
    )?;
    stable_files(raw, files)
}

fn raw_snapshot(raw: &Path) -> Result<BTreeMap<String, Identity>> {
    let files = raw_paths(raw)?
        .into_iter()
        .map(|name| Ok((name.clone(), identity(&raw.join(name))?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    stable_raw(raw, &files)?;
    Ok(files)
}

fn verify_raw(
    raw: &Path,
    expected: &BTreeMap<String, ContentIdentity>,
) -> Result<BTreeMap<String, Identity>> {
    require(
        raw_paths(raw)? == expected.keys().cloned().collect(),
        "raw source file manifest changed or is incomplete",
    )?;
    let files = verify_files(raw, expected)?;
    stable_raw(raw, &files)?;
    Ok(files)
}

fn expected_tables(raw: &Path, receipt: &SourceReceipt) -> Result<BTreeSet<String>> {
    let mut tables = BTreeSet::new();
    for name in receipt.raw_files.keys() {
        let parts: Vec<_> = name.split('/').collect();
        match parts.as_slice() {
            ["db", file] => {
                tables.insert(format!("{}:Db", file.strip_suffix(".parquet").unwrap()));
            }
            ["tasks", table, "manifest.yaml"] if !receipt.preprocess.skip_tasks => {
                let task: TaskSpec = read_yaml(&raw.join(name))?;
                if task.task_type.as_deref() == Some("recommendation") {
                    continue;
                }
                for (file, split) in [("train", "Train"), ("val", "Val"), ("test", "Test")] {
                    if receipt
                        .raw_files
                        .contains_key(&format!("tasks/{table}/{file}.parquet"))
                    {
                        tables.insert(format!("{table}:{split}"));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(tables)
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    format_version: u32,
    encoding: String,
    num_features: usize,
    num_nodes: usize,
    num_datetime_cells: usize,
    base_archive: BTreeMap<String, Identity>,
    raw_dir: String,
    source_columns: BTreeMap<String, String>,
    normalization: Normalization,
    source_receipt: Identity,
    payload: Identity,
    binding: String,
}

fn read_metadata(pre: &Path) -> Result<Metadata> {
    let path = pre.join(META);
    let value: serde_json::Value = serde_json::from_reader(BufReader::new(File::open(&path).map_err(|e|
        std::io::Error::other(format!("exact_datetime requires {}: {e}; run admitted backfill_datetime or fresh exact preprocessing", path.display())))?))?;
    require(
        value["format_version"] == 2,
        "exact datetime requires admitted format_version=2; v1/current raw hashes cannot prove historical provenance",
    )?;
    let meta: Metadata = serde_json::from_value(value)?;
    require(
        meta.encoding == ENCODING && meta.num_features == FEATURES,
        "unsupported exact datetime encoding",
    )?;
    meta.normalization.validate()?;
    require(
        meta.base_archive
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            == BASE_FILES.iter().copied().collect(),
        "incomplete base archive identity",
    )?;
    validate_content_ids(&content_map(&meta.base_archive))?;
    Ok(meta)
}

fn check_receipt(pre: &Path, meta: &Metadata) -> Result<SourceReceipt> {
    let bytes = fs::read(pre.join(SOURCE))?;
    require(
        bytes.len() as u64 == meta.source_receipt.bytes
            && digest(&bytes) == meta.source_receipt.sha256,
        "source receipt content mismatch",
    )?;
    let receipt: SourceReceipt = serde_json::from_slice(&bytes)?;
    receipt.validate()?;
    require(
        content_map(&meta.base_archive) == receipt.base_archive,
        "sidecar/archive provenance binding mismatch",
    )?;
    Ok(receipt)
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

pub struct Sidecar {
    mmap: Mmap,
    pub normalization: Normalization,
    num_nodes: usize,
    count: usize,
    seconds_start: usize,
    nanos_start: usize,
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn layout(nodes: usize, count: usize) -> Result<(usize, usize, usize)> {
    let seconds = nodes
        .checked_add(1)
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_add(HEADER));
    let nanos = seconds.and_then(|n| count.checked_mul(8).and_then(|m| n.checked_add(m)));
    let end = nanos.and_then(|n| count.checked_mul(4).and_then(|m| n.checked_add(m)));
    match (seconds, nanos, end) {
        (Some(a), Some(b), Some(c)) => Ok((a, b, c)),
        _ => Err(std::io::Error::other("datetime sidecar size overflow").into()),
    }
}

impl Sidecar {
    pub fn open(pre_dir: &Path) -> Result<Self> {
        let meta = read_metadata(pre_dir)?;
        check_identity(&pre_dir.join(SOURCE), &meta.source_receipt)?;
        check_receipt(pre_dir, &meta)?;
        for (name, expected) in &meta.base_archive {
            check_identity(&pre_dir.join(name), expected)?;
        }
        check_identity(&pre_dir.join(DATA), &meta.payload)?;
        Self::map(pre_dir, &meta)
    }

    fn map(pre_dir: &Path, meta: &Metadata) -> Result<Self> {
        let path = pre_dir.join(DATA);
        let file = File::open(path)?;
        // Sidecars must remain immutable while a sampler holds the mapping.
        let mmap = unsafe { Mmap::map(&file)? };
        let (seconds_start, nanos_start, end) = layout(meta.num_nodes, meta.num_datetime_cells)?;
        require(
            mmap.len() == end && mmap.len() >= HEADER,
            "invalid exact datetime payload length",
        )?;
        require(
            &mmap[..8] == MAGIC
                && u64_at(&mmap, 8) == meta.num_nodes as u64
                && u64_at(&mmap, 16) == meta.num_datetime_cells as u64
                && u64_at(&mmap, 24) == HEADER as u64
                && u64_at(&mmap, 32) == seconds_start as u64
                && u64_at(&mmap, 40) == nanos_start as u64
                && u64_at(&mmap, 48) == end as u64
                && u64_at(&mmap, 56) == 0,
            "invalid exact datetime header",
        )?;
        require(
            u64_at(&mmap, HEADER) == 0
                && u64_at(&mmap, HEADER + meta.num_nodes * 8) == meta.num_datetime_cells as u64,
            "invalid exact datetime terminal offsets",
        )?;
        Ok(Self {
            mmap,
            normalization: meta.normalization.clone(),
            num_nodes: meta.num_nodes,
            count: meta.num_datetime_cells,
            seconds_start,
            nanos_start,
        })
    }

    pub fn value(&self, node: &ArchivedNode, cell: usize) -> Result<Timestamp> {
        let idx = i32::from(node.node_idx);
        require(
            idx >= 0 && (idx as usize) < self.num_nodes,
            "sidecar node index out of range",
        )?;
        require(
            cell < node.sem_types.len()
                && matches!(node.sem_types[cell], ArchivedSemType::DateTime),
            "exact datetime requested for a non-datetime cell",
        )?;
        let idx = idx as usize;
        let start = u64_at(&self.mmap, HEADER + idx * 8) as usize;
        let end = u64_at(&self.mmap, HEADER + (idx + 1) * 8) as usize;
        let rank = node.sem_types[..cell]
            .iter()
            .filter(|s| matches!(s, ArchivedSemType::DateTime))
            .count();
        require(
            start <= end && end <= self.count && rank < end - start,
            "exact datetime node/cell offset mismatch",
        )?;
        let i = start + rank;
        let seconds = i64::from_le_bytes(
            self.mmap[self.seconds_start + i * 8..self.seconds_start + (i + 1) * 8]
                .try_into()
                .unwrap(),
        );
        let nanoseconds = u32::from_le_bytes(
            self.mmap[self.nanos_start + i * 4..self.nanos_start + (i + 1) * 4]
                .try_into()
                .unwrap(),
        );
        require(nanoseconds < 1_000_000_000, "invalid sidecar nanoseconds")?;
        Ok(Timestamp {
            seconds,
            nanoseconds,
        })
    }
}

#[derive(Default, Deserialize)]
struct TableSpec {
    pkey: Option<String>,
    time_col: Option<String>,
    #[serde(default)]
    fkeys: HashMap<String, String>,
}
#[derive(Deserialize)]
struct Manifest {
    name: String,
    #[serde(default)]
    tables: HashMap<String, TableSpec>,
    val_timestamp: Option<String>,
}
#[derive(Deserialize)]
struct TaskSpec {
    task_type: Option<String>,
    time_col: Option<String>,
    entity_table: Option<String>,
    entity_col: Option<String>,
    src_entity_table: Option<String>,
    src_entity_col: Option<String>,
    dst_entity_table: Option<String>,
    dst_entity_col: Option<String>,
}

fn read_yaml<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_yaml::from_reader(BufReader::new(File::open(path)?))?)
}

fn archive_node<'a>(
    nodes: &'a [u8],
    offsets: &ArchivedOffsets,
    idx: usize,
) -> Result<&'a ArchivedNode> {
    let next = idx
        .checked_add(1)
        .ok_or_else(|| std::io::Error::other("archive node index overflow"))?;
    require(
        next < offsets.offsets.len(),
        format!("archive node index out of range: {idx}"),
    )?;
    let (left, right) = (
        i64::from(offsets.offsets[idx]),
        i64::from(offsets.offsets[next]),
    );
    require(
        left >= 0 && right >= left && right as usize <= nodes.len(),
        format!("invalid node archive offsets at node {idx}"),
    )?;
    let node = rkyv::access::<ArchivedNode, Error>(&nodes[left as usize..right as usize])?;
    let n = node.col_name_idxs.len();
    require(
        node.sem_types.len() == n && node.datetime_values.len() == n,
        format!("parallel cell arrays mismatch at node {idx}"),
    )?;
    require(
        n == 0 || i32::from(node.node_idx) == i32::try_from(idx)?,
        "archived node identity mismatch",
    )?;
    Ok(node)
}

fn integer(v: AnyValue<'_>) -> Result<Option<i64>> {
    match v {
        AnyValue::Null => Ok(None),
        AnyValue::Int8(v) => Ok(Some(v.into())),
        AnyValue::Int16(v) => Ok(Some(v.into())),
        AnyValue::Int32(v) => Ok(Some(v.into())),
        AnyValue::Int64(v) => Ok(Some(v)),
        AnyValue::UInt8(v) => Ok(Some(v.into())),
        AnyValue::UInt16(v) => Ok(Some(v.into())),
        AnyValue::UInt32(v) => Ok(Some(v.into())),
        AnyValue::UInt64(v) => Ok(Some(i64::try_from(v)?)),
        AnyValue::Float64(v) if v.is_nan() => Ok(None),
        AnyValue::Float64(v)
            if v.is_finite()
                && v.fract() == 0.0
                && v >= i64::MIN as f64
                && v < 9_223_372_036_854_775_808.0 =>
        {
            Ok(Some(v as i64))
        }
        AnyValue::Float32(v) if v.is_nan() => Ok(None),
        AnyValue::Float32(v)
            if v.is_finite()
                && v.fract() == 0.0
                && v >= i64::MIN as f32
                && v < 9_223_372_036_854_775_808.0_f32 =>
        {
            Ok(Some(v as i64))
        }
        v => Err(std::io::Error::other(format!("invalid row key {v:?}")).into()),
    }
}

fn foreign_values(v: AnyValue<'_>) -> Result<Vec<i64>> {
    match v {
        AnyValue::List(values) => {
            let values: Result<Vec<Option<i64>>> = values.iter().map(integer).collect();
            Ok(values?.into_iter().flatten().collect())
        }
        v => Ok(integer(v)?.into_iter().collect()),
    }
}

struct Scratch {
    path: PathBuf,
    lock: PathBuf,
}
impl Scratch {
    fn new(pre: &Path) -> Result<Self> {
        let lock = pre.join(".datetime_exact.lock");
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock)
            .map_err(|e| {
                std::io::Error::other(format!("cannot acquire {}: {e}", lock.display()))
            })?;
        let suffix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = pre.join(format!(".datetime_exact-{}-{suffix}", std::process::id()));
        if let Err(e) = fs::create_dir(&path) {
            let _ = fs::remove_file(&lock);
            return Err(e.into());
        }
        Ok(Self { path, lock })
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
        let _ = fs::remove_file(&self.lock);
    }
}

/// Observe the source around the actual legacy preprocessing invocation. No
/// historical receipt is manufactured by this entry point.
#[allow(dead_code)]
pub fn preprocess_exact(cli: crate::pre::Cli) -> Result<()> {
    let raw = PathBuf::from(&cli.dataset_dir);
    let before = raw_snapshot(&raw)?;
    let manifest: Manifest = read_yaml(&raw.join("manifest.yaml"))?;
    require(normal_name(&manifest.name), "invalid dataset name")?;
    let pre = Path::new(&cli.out_dir).join(&manifest.name);
    fs::create_dir_all(&pre)?;
    let scratch = Scratch::new(&pre)?;
    let policy = std::env::var("RT_IDENTIFIER_POLICY").unwrap_or_else(|_| "threshold".into());
    require(
        matches!(
            policy.as_str(),
            "none" | "empty" | "empty_or_time" | "threshold"
        ),
        "invalid RT_IDENTIFIER_POLICY",
    )?;
    let config = PreprocessConfig {
        implementation: format!(
            "pre.rs+common.rs:sha256:{}",
            digest(
                &[
                    include_bytes!("pre.rs").as_slice(),
                    include_bytes!("common.rs").as_slice()
                ]
                .concat()
            )
        ),
        skip_tasks: cli.skip_tasks,
        skip_db: cli.skip_db,
        identifier_policy: policy.clone(),
        raw_manifest_policy: RAW_POLICY.into(),
    };
    // An interrupted rebuild must never leave an earlier receipt admissible.
    for name in [META, SOURCE] {
        match fs::remove_file(pre.join(name)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    std::panic::catch_unwind(|| crate::pre::main(cli)).map_err(|e| {
        let detail = e
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| e.downcast_ref::<&str>().copied())
            .unwrap_or("unknown panic");
        std::io::Error::other(format!("fresh preprocessing failed: {detail}"))
    })?;
    let raw_files = content_map(&before);
    let after = verify_raw(&raw, &raw_files)?;
    require(
        std::env::var("RT_IDENTIFIER_POLICY").unwrap_or_else(|_| "threshold".into()) == policy,
        "identifier policy changed during preprocessing",
    )?;
    let base = base_identities(&pre)?;
    let archive = content_map(&base);
    let receipt = SourceReceipt {
        format_version: 1, dataset_name: manifest.name, kind: "fresh_preprocess".into(),
        preprocess: config,
        origin: Origin {
            issuer: "rt.rustler.preprocess".into(),
            raw_revision: format!("sha256:{}", digest(&serde_json::to_vec(&raw_files)?)),
            archive_revision: format!("sha256:{}", digest(&serde_json::to_vec(&archive)?)),
            evidence: "Observed native preprocessing between matching complete raw input inventories; not publisher authentication".into(),
        }, raw_files, base_archive: archive,
    };
    receipt.validate()?;
    stable_raw(&raw, &after)?;
    let bytes = serde_json::to_vec_pretty(&receipt)?;
    build_sidecar(&raw, &pre, &scratch, &receipt, &bytes, base)
}

/// An external receipt is a caller-supplied trust assertion, not a claim that
/// current raw bytes or a lossy replay prove the archive's historical source.
#[allow(dead_code)]
pub fn backfill(raw_dir: &Path, pre_dir: &Path, source_receipt: Option<&Path>) -> Result<()> {
    let scratch = Scratch::new(pre_dir)?;
    let stored = pre_dir.join(SOURCE);
    let path = source_receipt.unwrap_or(&stored);
    let state = file_state(path).map_err(|e| std::io::Error::other(format!(
        "missing source receipt {}: {e}; historical backfill requires an independently trusted source_receipt", path.display())))?;
    let bytes = fs::read(path)?;
    require(
        file_state(path)? == state,
        "source receipt changed while reading",
    )?;
    let receipt: SourceReceipt = serde_json::from_slice(&bytes)?;
    receipt.validate()?;
    require(
        source_receipt.is_some() || receipt.kind == "fresh_preprocess",
        "historical backfill requires an explicit independently trusted source_receipt path",
    )?;
    let raw = verify_raw(raw_dir, &receipt.raw_files)?;
    let base = verify_files(pre_dir, &receipt.base_archive)?;
    stable_raw(raw_dir, &raw)?;
    build_sidecar(raw_dir, pre_dir, &scratch, &receipt, &bytes, base)
}

fn table_end(info: &TableInfo, total: usize) -> Result<usize> {
    require(
        info.node_idx_offset >= 0 && info.num_nodes >= 0,
        "negative table range",
    )?;
    let end = (info.node_idx_offset as usize)
        .checked_add(info.num_nodes as usize)
        .ok_or_else(|| std::io::Error::other("table range overflow"))?;
    require(end <= total, "table range exceeds archive node count")?;
    Ok(end)
}

/// Build only after source admission, without changing Node or text embeddings.
fn build_sidecar(
    raw_dir: &Path,
    pre_dir: &Path,
    scratch: &Scratch,
    receipt: &SourceReceipt,
    receipt_bytes: &[u8],
    base_archive: BTreeMap<String, Identity>,
) -> Result<()> {
    let manifest_path = raw_dir.join("manifest.yaml");
    let manifest: Manifest = read_yaml(&manifest_path)?;
    let legacy_meta: serde_json::Value =
        serde_json::from_reader(File::open(pre_dir.join("meta.json"))?)?;
    require(
        legacy_meta["format_version"] == 1
            && legacy_meta["name"] == manifest.name
            && manifest.name == receipt.dataset_name,
        "raw dataset name or archive format mismatch",
    )?;
    let tables: HashMap<String, TableInfo> =
        serde_json::from_reader(File::open(pre_dir.join("table_info.json"))?)?;
    let column_index: HashMap<String, i32> =
        serde_json::from_reader(File::open(pre_dir.join("column_index.json"))?)?;
    require(
        tables.keys().cloned().collect::<BTreeSet<_>>() == expected_tables(raw_dir, receipt)?,
        "archive table inventory does not match source receipt preprocessing policy",
    )?;
    let nodes = unsafe { Mmap::map(&File::open(pre_dir.join("nodes.rkyv"))?)? };
    let offsets_map = unsafe { Mmap::map(&File::open(pre_dir.join("offsets.rkyv"))?)? };
    // Keep the original offsets mapped too: a large corpus must not allocate
    // another eight bytes per node just to build the sidecar.
    let offsets = rkyv::access::<ArchivedOffsets, Error>(&offsets_map)?;
    require(!offsets.offsets.is_empty(), "missing archive offsets")?;
    let n = offsets.offsets.len() - 1;
    require(
        i32::try_from(n).is_ok(),
        "archive node count exceeds supported index range",
    )?;
    require(
        legacy_meta["num_nodes"].as_u64() == Some(n as u64)
            && i64::from(offsets.offsets[0]) == 0
            && i64::from(offsets.offsets[n]) == nodes.len() as i64,
        "archive node count/terminal offset mismatch",
    )?;

    let offsets_path = scratch.path.join("offsets");
    let mut out_offsets = BufWriter::new(File::create(&offsets_path)?);
    let mut count = 0_usize;
    out_offsets.write_all(&0_u64.to_le_bytes())?;
    for idx in 0..n {
        let node = archive_node(&nodes, offsets, idx)?;
        let mut seen = HashSet::new();
        for (col, sem) in node.col_name_idxs.iter().zip(node.sem_types.iter()) {
            require(
                seen.insert(i32::from(*col)),
                format!("duplicate column identity at node {idx}"),
            )?;
            count = count
                .checked_add(usize::from(matches!(sem, ArchivedSemType::DateTime)))
                .ok_or_else(|| std::io::Error::other("datetime cell count overflow"))?;
        }
        out_offsets.write_all(&(count as u64).to_le_bytes())?;
    }
    out_offsets.flush()?;
    let (seconds_start, nanos_start, end) = layout(n, count)?;
    let payload_path = scratch.path.join(DATA);
    let mut file = File::create(&payload_path)?;
    file.write_all(MAGIC)?;
    for v in [n, count, HEADER, seconds_start, nanos_start, end, 0] {
        file.write_all(&(v as u64).to_le_bytes())?;
    }
    std::io::copy(&mut File::open(offsets_path)?, &mut file)?;
    file.set_len(end as u64)?;
    drop(file);
    let mut sec_file = OpenOptions::new().write(true).open(&payload_path)?;
    sec_file.seek(SeekFrom::Start(seconds_start as u64))?;
    let mut ns_file = OpenOptions::new().write(true).open(&payload_path)?;
    ns_file.seek(SeekFrom::Start(nanos_start as u64))?;
    let mut seconds = BufWriter::new(sec_file);
    let mut nanos = BufWriter::new(ns_file);
    let mut source_columns = BTreeMap::new();
    let cutoff = manifest
        .val_timestamp
        .as_deref()
        .map(|s| -> Result<Timestamp> {
            let dt = NaiveDate::parse_from_str(s, "%Y-%m-%d")?
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc();
            Ok(Timestamp {
                seconds: dt.timestamp(),
                nanoseconds: 0,
            })
        })
        .transpose()?;
    let mut ordered: Vec<_> = tables.iter().collect();
    ordered.sort_by_key(|(name, info)| (info.node_idx_offset, info.num_nodes > 0, *name));
    let mut next_node = 0;
    let mut written = 0;
    let mut moments = Moments::default();
    for (key, info) in ordered {
        let end_node = table_end(info, n)?;
        require(
            info.node_idx_offset as usize == next_node,
            format!("noncontiguous table range: {key}"),
        )?;
        let (table, split) = key
            .rsplit_once(':')
            .ok_or_else(|| std::io::Error::other(format!("invalid table key {key}")))?;
        require(normal_name(table), "invalid raw table name")?;
        let (relative, spec) = if split == "Db" {
            (
                format!("db/{table}.parquet"),
                manifest
                    .tables
                    .get(table)
                    .map(|s| TableSpec {
                        pkey: s.pkey.clone(),
                        time_col: s.time_col.clone(),
                        fkeys: s.fkeys.clone(),
                    })
                    .unwrap_or_default(),
            )
        } else {
            let stem = match split {
                "Train" => "train",
                "Val" => "val",
                "Test" => "test",
                _ => return Err(std::io::Error::other(format!("invalid split {split}")).into()),
            };
            let task_manifest = format!("tasks/{table}/manifest.yaml");
            let task: TaskSpec = read_yaml(&raw_dir.join(&task_manifest))?;
            let mut fkeys = HashMap::new();
            for (col, parent) in [
                (task.entity_col, task.entity_table),
                (task.src_entity_col, task.src_entity_table),
                (task.dst_entity_col, task.dst_entity_table),
            ] {
                if let (Some(col), Some(parent)) = (col, parent) {
                    fkeys.insert(col, parent);
                }
            }
            (
                format!("tasks/{table}/{stem}.parquet"),
                TableSpec {
                    pkey: None,
                    time_col: task.time_col,
                    fkeys,
                },
            )
        };
        let path = raw_dir.join(&relative);
        let mut reader = ParquetReader::new(File::open(&path)?);
        require(
            reader.num_rows()? == info.num_nodes as usize,
            format!("raw row-count mismatch: {relative}"),
        )?;
        if split == "Db" && receipt.preprocess.skip_db {
            for idx in next_node..end_node {
                let node = archive_node(&nodes, offsets, idx)?;
                require(
                    node.col_name_idxs.is_empty() && node.f2p_nbr_idxs.is_empty(),
                    "skip_db receipt requires blank archived DB nodes",
                )?;
            }
            next_node = end_node;
            continue;
        }
        let parquet_metadata = reader.get_metadata()?.clone();
        let schema = reader.with_slice(Some((0, 0))).finish()?;
        let mut selected = Vec::new();
        let mut date_columns = HashMap::new();
        for column in schema.iter() {
            let name = column.name().as_str();
            let is_date = matches!(column.dtype(), DataType::Date | DataType::Datetime(_, _));
            if is_date {
                source_columns.insert(format!("{key}/{name}"), format!("{:?}", column.dtype()));
                if let Some(&id) = column_index.get(&format!("{name} of {table}")) {
                    require(
                        date_columns.insert(id, name.to_owned()).is_none(),
                        "ambiguous source column id",
                    )?;
                }
            }
            if is_date || spec.pkey.as_deref() == Some(name) || spec.fkeys.contains_key(name) {
                selected.push(name.to_owned());
            }
        }
        // A zero-date/key table needs only row counts; no large text projection.
        for start in (0..info.num_nodes as usize).step_by(65_536) {
            let len = (info.num_nodes as usize - start).min(65_536);
            let df = if selected.is_empty() {
                None
            } else {
                let mut reader = ParquetReader::new(File::open(&path)?);
                reader.set_metadata(parquet_metadata.clone());
                Some(
                    reader
                        .with_columns(Some(selected.clone()))
                        .with_slice(Some((start, len)))
                        .finish()?,
                )
            };
            if let Some(df) = &df {
                require(df.height() == len, "short raw Parquet batch")?;
            }
            for r in 0..len {
                let row = start + r;
                let idx = info.node_idx_offset as usize + row;
                let node = archive_node(&nodes, offsets, idx)?;
                require(
                    node.col_name_idxs.is_empty() || node.is_task_node == (split != "Db"),
                    format!("archive split mismatch: {key} row {row}"),
                )?;
                if let (Some(pk), Some(df)) = (&spec.pkey, &df) {
                    require(
                        integer(df.column(pk)?.get(r)?)? == Some(row as i64),
                        format!("raw primary-key row order mismatch: {key} row {row}"),
                    )?;
                }
                let mut raw_parents = Vec::new();
                if let Some(df) = &df {
                    for (fk, parent) in &spec.fkeys {
                        let col = df.column(fk)?;
                        if matches!(col.dtype(), DataType::Date | DataType::Datetime(_, _)) {
                            continue;
                        }
                        let parent_info = tables.get(&format!("{parent}:Db")).ok_or_else(|| {
                            std::io::Error::other(format!("missing parent table {parent}"))
                        })?;
                        for id in foreign_values(col.get(r)?)? {
                            require(
                                id >= 0 && id < i64::from(parent_info.num_nodes),
                                format!("foreign key outside parent: {key} row {row}"),
                            )?;
                            raw_parents.push(i64::from(parent_info.node_idx_offset) + id);
                        }
                    }
                }
                let mut archived_parents: Vec<_> = node
                    .f2p_nbr_idxs
                    .iter()
                    .map(|x| i64::from(i32::from(*x)))
                    .collect();
                raw_parents.sort_unstable();
                archived_parents.sort_unstable();
                require(
                    raw_parents == archived_parents,
                    format!("foreign-key row binding mismatch: {key} row {row}"),
                )?;
                let eligible = match (split, cutoff, &spec.time_col, &df) {
                    ("Val" | "Test", _, _, _) => false,
                    ("Db", Some(cutoff), Some(time), Some(df)) => {
                        Timestamp::from_value(df.column(time)?.get(r)?)?
                            .is_some_and(|t| t <= cutoff)
                    }
                    _ => true,
                };
                // Every retained temporal column must have exactly the source
                // null mask. Keys and dropped constant columns are not cells.
                for (&id, col) in &date_columns {
                    if spec.pkey.as_deref() == Some(col.as_str()) || spec.fkeys.contains_key(col) {
                        continue;
                    }
                    let df = df
                        .as_ref()
                        .ok_or_else(|| std::io::Error::other("missing temporal projection"))?;
                    let raw_valid = Timestamp::from_value(df.column(col)?.get(r)?)?.is_some();
                    let archived_valid =
                        node.col_name_idxs
                            .iter()
                            .zip(node.sem_types.iter())
                            .any(|(c, s)| {
                                i32::from(*c) == id && matches!(s, ArchivedSemType::DateTime)
                            });
                    require(
                        raw_valid == archived_valid,
                        format!(
                            "datetime null/cell binding mismatch: {key} row {row} column {col}"
                        ),
                    )?;
                }
                for (cell, sem) in node.sem_types.iter().enumerate() {
                    if !matches!(sem, ArchivedSemType::DateTime) {
                        continue;
                    }
                    let col = date_columns
                        .get(&i32::from(node.col_name_idxs[cell]))
                        .ok_or_else(|| {
                            std::io::Error::other(format!(
                                "unmatched datetime column: {key} row {row} cell {cell}"
                            ))
                        })?;
                    let df = df
                        .as_ref()
                        .ok_or_else(|| std::io::Error::other("missing temporal projection"))?;
                    let timestamp =
                        Timestamp::from_value(df.column(col)?.get(r)?)?.ok_or_else(|| {
                            std::io::Error::other(format!(
                                "raw null for archived datetime: {key} row {row} column {col}"
                            ))
                        })?;
                    timestamp.features()?; // Validate calendar range before publishing.
                    seconds.write_all(&timestamp.seconds.to_le_bytes())?;
                    nanos.write_all(&timestamp.nanoseconds.to_le_bytes())?;
                    written += 1;
                    if eligible {
                        moments.push(timestamp);
                    }
                }
                next_node += 1;
            }
        }
    }
    require(
        next_node == n && written == count,
        "incomplete datetime node/cell coverage",
    )?;
    seconds.flush()?;
    nanos.flush()?;
    seconds.get_ref().sync_all()?;
    nanos.get_ref().sync_all()?;
    drop(seconds);
    drop(nanos);
    // A complete second source scan detects persistent changes while reading.
    // The archive stays immutable; fast state checks detect concurrent rewrites.
    let raw_after = verify_raw(raw_dir, &receipt.raw_files)?;
    stable_files(pre_dir, &base_archive)?;
    stable_raw(raw_dir, &raw_after)?;
    let normalization = moments.finish();
    normalization.validate()?;
    let source_path = scratch.path.join(SOURCE);
    let mut source_file = File::create(&source_path)?;
    source_file.write_all(receipt_bytes)?;
    source_file.sync_all()?;
    let metadata = Metadata {
        format_version: 2, encoding: ENCODING.into(), num_features: FEATURES,
        num_nodes: n, num_datetime_cells: count, base_archive,
        raw_dir: raw_dir.display().to_string(), source_columns, normalization,
        source_receipt: identity(&source_path)?, payload: identity(&payload_path)?,
        binding: "admitted source receipt + complete SHA-256 input/archive commitments + positional node/cell checks; historical authority is caller-trusted, not hash-authenticated".into(),
    };
    write_json(&scratch.path.join(META), &metadata)?;
    stable_files(pre_dir, &metadata.base_archive)?;
    stable_raw(raw_dir, &raw_after)?;
    fs::rename(source_path, pre_dir.join(SOURCE))?;
    fs::rename(payload_path, pre_dir.join(DATA))?;
    fs::rename(scratch.path.join(META), pre_dir.join(META))?;
    File::open(pre_dir)?.sync_all()?;
    Ok(())
}

/// Full portable validation, followed only on success by an atomic local stat
/// receipt refresh. No raw inputs or historical provenance are manufactured.
#[allow(dead_code)]
pub fn validate(pre: &Path) -> Result<()> {
    let scratch = Scratch::new(pre)?;
    let mut meta = read_metadata(pre)?;
    let receipt_id = identity(&pre.join(SOURCE))?;
    require(
        receipt_id.content() == meta.source_receipt.content(),
        "source receipt content mismatch",
    )?;
    check_receipt(pre, &meta)?;
    let base = verify_files(pre, &content_map(&meta.base_archive))?;
    let payload = identity(&pre.join(DATA))?;
    require(
        payload.content() == meta.payload.content(),
        "datetime payload content mismatch",
    )?;
    let sidecar = Sidecar::map(pre, &meta)?;
    let nodes_file = File::open(pre.join("nodes.rkyv"))?;
    let nodes = if nodes_file.metadata()?.len() == 0 {
        None
    } else {
        Some(unsafe { Mmap::map(&nodes_file)? })
    };
    let nodes = nodes.as_deref().unwrap_or(&[]);
    let offsets_map = unsafe { Mmap::map(&File::open(pre.join("offsets.rkyv"))?)? };
    let offsets = rkyv::access::<ArchivedOffsets, Error>(&offsets_map)?;
    require(
        meta.num_nodes < offsets.offsets.len()
            && offsets.offsets.len() - 1 == meta.num_nodes
            && i32::try_from(meta.num_nodes).is_ok()
            && i64::from(offsets.offsets[0]) == 0
            && i64::from(offsets.offsets[meta.num_nodes]) == nodes.len() as i64,
        "archive node count/terminal offset mismatch",
    )?;
    let tables: HashMap<String, TableInfo> =
        serde_json::from_reader(File::open(pre.join("table_info.json"))?)?;
    let mut ranges: Vec<_> = tables.values().collect();
    ranges.sort_by_key(|info| (info.node_idx_offset, info.num_nodes > 0));
    let mut next_node = 0;
    for info in ranges {
        require(
            info.node_idx_offset as usize == next_node,
            "noncontiguous archive table range",
        )?;
        next_node = table_end(info, meta.num_nodes)?;
    }
    require(
        next_node == meta.num_nodes,
        "incomplete archive table coverage",
    )?;
    let mut count = 0usize;
    for idx in 0..meta.num_nodes {
        let node = archive_node(nodes, offsets, idx)?;
        require(
            u64_at(&sidecar.mmap, HEADER + idx * 8) == count as u64,
            "datetime node/cell offset mismatch",
        )?;
        for (cell, sem) in node.sem_types.iter().enumerate() {
            if matches!(sem, ArchivedSemType::DateTime) {
                sidecar.value(node, cell)?.features()?;
                count = count
                    .checked_add(1)
                    .ok_or_else(|| std::io::Error::other("datetime cell count overflow"))?;
            }
        }
        require(
            u64_at(&sidecar.mmap, HEADER + (idx + 1) * 8) == count as u64,
            "datetime node/cell offset mismatch",
        )?;
    }
    require(
        count == meta.num_datetime_cells,
        "datetime cell count mismatch",
    )?;
    stable_files(pre, &base)?;
    require(
        file_state(&pre.join(DATA))? == (payload.bytes, payload.modified_ns)
            && file_state(&pre.join(SOURCE))? == (receipt_id.bytes, receipt_id.modified_ns),
        "sidecar changed during validation",
    )?;
    meta.base_archive = base;
    meta.payload = payload;
    meta.source_receipt = receipt_id;
    write_json(&scratch.path.join(META), &meta)?;
    fs::rename(scratch.path.join(META), pre.join(META))?;
    File::open(pre)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_units_and_negative_remainders() {
        for (unit, q) in [
            (TimeUnit::Milliseconds, 1000_i64),
            (TimeUnit::Microseconds, 1_000_000),
            (TimeUnit::Nanoseconds, 1_000_000_000),
        ] {
            for value in [i64::MIN, -1, 0, 1, i64::MAX] {
                let t = Timestamp::from_ticks(value, unit);
                assert!(t.nanoseconds < 1_000_000_000);
                let restored = i128::from(t.seconds) * i128::from(q)
                    + i128::from(t.nanoseconds) / i128::from(1_000_000_000 / q);
                assert_eq!(restored, i128::from(value));
            }
        }
        assert_eq!(
            Timestamp::from_ticks(-1, TimeUnit::Nanoseconds),
            Timestamp {
                seconds: -1,
                nanoseconds: 999_999_999
            }
        );
    }

    #[test]
    fn calendar14_bf16_roundtrip_and_range() {
        for year in [
            NaiveDate::MIN.year(),
            -1000,
            -1,
            0,
            1600,
            1900,
            2000,
            2024,
            2263,
            9999,
            NaiveDate::MAX.year(),
        ] {
            let dt = NaiveDate::from_ymd_opt(year, 3, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc();
            for ns in [0, 1, 255, 256, 999, 1000, 999_999, 1_000_000, 999_999_999] {
                let t = Timestamp {
                    seconds: dt.timestamp(),
                    nanoseconds: ns,
                };
                let v = t.features().unwrap().map(|x| (x.to_f32() * 256.0) as i32);
                assert_eq!(v[0] * 65536 + v[1] * 256 + v[2], year);
                assert_eq!(&v[3..8], &[3, 1, 0, 0, 0]);
                let ns2 = (v[8] * 256 + v[9]) * 1_000_000
                    + (v[10] * 256 + v[11]) * 1_000
                    + v[12] * 256
                    + v[13];
                assert_eq!(ns2 as u32, ns);
            }
        }
        assert!(
            Timestamp {
                seconds: i64::MAX,
                nanoseconds: 0
            }
            .features()
            .is_err()
        );
        assert!(
            Timestamp {
                seconds: 0,
                nanoseconds: 1_000_000_000
            }
            .features()
            .is_err()
        );
    }

    #[test]
    fn normalization_origin_population_and_constant_fallback() {
        let origin = Timestamp {
            seconds: 253_402_300_799,
            nanoseconds: 123,
        };
        let next = Timestamp {
            nanoseconds: 125,
            ..origin
        };
        let mut moments = Moments::default();
        moments.push(origin);
        moments.push(next);
        let norm = moments.finish();
        assert!((norm.std_seconds - 1e-9).abs() < 1e-24);
        assert_eq!(norm.target(origin), -1.0);
        assert_eq!(norm.target(next), 1.0);
        let mut constant = Moments::default();
        constant.push(origin);
        constant.push(origin);
        let norm = constant.finish();
        assert_eq!(norm.std_seconds, 1.0);
        assert_eq!(norm.target(origin), 0.0);
        assert_eq!(Moments::default().finish().std_seconds, 1.0);
    }
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "rt-datetime-test-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn parquet_dates(path: &Path, dates: &[Option<i64>], primary_key: bool) {
        let len = dates.len();
        let dates = Series::new("event".into(), dates)
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let mut cols = vec![dates.into()];
        if primary_key {
            cols.push(
                Series::new("id".into(), (0..len).map(|v| v as i64).collect::<Vec<_>>()).into(),
            );
        }
        let mut df = DataFrame::new(cols).unwrap();
        ParquetWriter::new(File::create(path).unwrap())
            .finish(&mut df)
            .unwrap();
    }

    fn sampler(
        pre: &Path,
        table: &str,
        exact: bool,
    ) -> std::result::Result<crate::fly::Sampler, String> {
        let tables: HashMap<String, TableInfo> =
            serde_json::from_reader(File::open(pre.join("table_info.json")).unwrap()).unwrap();
        let columns: HashMap<String, i32> =
            serde_json::from_reader(File::open(pre.join("column_index.json")).unwrap()).unwrap();
        let info = &tables[&format!("{table}:Db")];
        crate::fly::Sampler::new_impl(
            vec![(
                "fixture".into(),
                table.into(),
                info.node_idx_offset,
                info.num_nodes,
            )],
            0,
            0,
            1,
            vec![4],
            vec![4],
            0,
            10,
            vec![false],
            0.5,
            "fixture",
            pre.parent().unwrap().display().to_string(),
            1,
            0,
            0,
            vec![columns[&format!("event of {table}")]],
            vec![vec![]],
            vec![None],
            -1,
            true,
            true,
            0,
            false,
            false,
            5.0,
            None,
            exact,
        )
    }

    #[test]
    fn backfill_preserves_base_and_binds_cells_and_training_stats() {
        let temp = Temp::new();
        let raw = temp.0.join("raw");
        let output = temp.0.join("pre");
        fs::create_dir_all(raw.join("db")).unwrap();
        fs::create_dir_all(raw.join("tasks/forecast")).unwrap();
        fs::write(raw.join("manifest.yaml"),
            "name: fixture\nval_timestamp: '1970-01-01'\ntables:\n  events:\n    pkey: id\n    time_col: event\n").unwrap();
        fs::write(
            raw.join("tasks/forecast/manifest.yaml"),
            "target_col: event\ntask_type: regression\n",
        )
        .unwrap();
        // An empty table shares its offset with the following nonempty split.
        parquet_dates(&raw.join("db/z_empty.parquet"), &[], true);
        let source = [Some(-1), Some(0), Some(253_402_300_799_123_456), None];
        parquet_dates(&raw.join("db/events.parquet"), &source, true);
        parquet_dates(
            &raw.join("tasks/forecast/train.parquet"),
            &[Some(0), Some(2_000_000)],
            false,
        );
        parquet_dates(
            &raw.join("tasks/forecast/val.parquet"),
            &[Some(100_000_000)],
            false,
        );
        parquet_dates(
            &raw.join("tasks/forecast/test.parquet"),
            &[Some(200_000_000)],
            false,
        );
        preprocess_exact(crate::pre::Cli {
            dataset_dir: raw.display().to_string(),
            out_dir: output.display().to_string(),
            source: None,
            skip_tasks: false,
            skip_db: false,
        })
        .unwrap();
        let pre = output.join("fixture");
        let before: Vec<_> = BASE_FILES
            .iter()
            .map(|f| fs::read(pre.join(f)).unwrap())
            .collect();
        let legacy_meta = fs::read(pre.join("meta.json")).unwrap();
        let meta: serde_json::Value = serde_json::from_slice(&legacy_meta).unwrap();
        fs::write(
            pre.join("text_emb_fixture.bin"),
            vec![0; meta["num_text_strings"].as_u64().unwrap() as usize * 2],
        )
        .unwrap();
        fs::remove_file(pre.join(META)).unwrap();
        fs::remove_file(pre.join(DATA)).unwrap();
        assert!(Sidecar::open(&pre).is_err());
        // Missing exact data cannot be swallowed even with ignore_data_errors.
        assert!(
            sampler(&pre, "events", true)
                .err()
                .unwrap()
                .contains("backfill_datetime")
        );
        let legacy = sampler(&pre, "events", false)
            .unwrap()
            .batch(Some(0), 0, 4, 4);
        assert!(legacy.exact.is_none());
        backfill(&raw, &pre, None).unwrap();
        let exact = sampler(&pre, "events", true)
            .unwrap()
            .batch(Some(0), 0, 4, 4);
        assert_eq!(legacy.node_idxs, exact.node_idxs);
        assert_eq!(legacy.f2p_nbr_idxs, exact.f2p_nbr_idxs);
        assert_eq!(legacy.table_name_idxs, exact.table_name_idxs);
        assert_eq!(legacy.col_name_idxs, exact.col_name_idxs);
        assert_eq!(legacy.class_value_idxs, exact.class_value_idxs);
        assert_eq!(legacy.sem_types, exact.sem_types);
        assert_eq!(legacy.is_targets, exact.is_targets);
        assert_eq!(legacy.is_task_nodes, exact.is_task_nodes);
        assert_eq!(legacy.is_padding, exact.is_padding);
        assert_eq!(legacy.timestamps, exact.timestamps);
        assert_eq!(legacy.seed_node_idxs, exact.seed_node_idxs);
        assert_eq!(legacy.bfs_depths, exact.bfs_depths);
        assert_eq!(legacy.batch_mask, exact.batch_mask);
        assert_eq!(legacy.seq_len, exact.seq_len);
        assert_eq!(
            legacy
                .col_name_values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            exact
                .col_name_values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            legacy
                .number_values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            exact
                .number_values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            legacy
                .text_values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            exact
                .text_values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            legacy
                .datetime_values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            exact
                .datetime_values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let data = exact.exact.as_ref().unwrap();
        assert_eq!(data.features.len(), 4 * 4 * FEATURES);
        for i in 0..data.valid.len() {
            assert_eq!(
                data.valid[i],
                !exact.is_padding[i] && exact.sem_types[i] == 2
            );
            if data.valid[i] {
                let raw = Timestamp {
                    seconds: data.seconds[i],
                    nanoseconds: data.nanoseconds[i] as u32,
                };
                assert_eq!(
                    &data.features[i * FEATURES..(i + 1) * FEATURES],
                    &raw.features().unwrap()
                );
                assert!(data.target_values[i].is_finite());
            } else {
                assert_eq!(data.seconds[i], 0);
                assert_eq!(data.target_values[i], 0.0);
            }
        }
        for (name, bytes) in BASE_FILES.iter().zip(&before) {
            assert_eq!(&fs::read(pre.join(name)).unwrap(), bytes);
        }
        let sidecar = Sidecar::open(&pre).unwrap();
        assert_eq!(sidecar.normalization.count, 4); // Db <= cutoff and Train, never Val/Test.
        let norm = &sidecar.normalization;
        let absolute_mean = norm.origin_seconds as f64
            + norm.origin_nanoseconds as f64 * 1e-9
            + norm.mean_offset_seconds;
        assert!((absolute_mean - 0.499_999_75).abs() < 1e-12);
        let expected_std = [-0.000001_f64, 0.0, 0.0, 2.0]
            .iter()
            .map(|v| (v - absolute_mean).powi(2))
            .sum::<f64>()
            .sqrt()
            / 2.0;
        assert!((norm.std_seconds - expected_std).abs() < 1e-12);
        let nodes = unsafe { Mmap::map(&File::open(pre.join("nodes.rkyv")).unwrap()).unwrap() };
        let offsets_bytes = fs::read(pre.join("offsets.rkyv")).unwrap();
        let offsets = rkyv::access::<ArchivedOffsets, Error>(&offsets_bytes).unwrap();
        let tables: HashMap<String, TableInfo> =
            serde_json::from_reader(File::open(pre.join("table_info.json")).unwrap()).unwrap();
        let first = tables["events:Db"].node_idx_offset as usize;
        for (row, value) in source.iter().enumerate() {
            let node = archive_node(&nodes, offsets, first + row).unwrap();
            let cell = node
                .sem_types
                .iter()
                .position(|s| matches!(s, ArchivedSemType::DateTime));
            if let Some(raw) = value {
                let ts = sidecar.value(node, cell.unwrap()).unwrap();
                assert_eq!(ts, Timestamp::from_ticks(*raw, TimeUnit::Microseconds));
                assert!(sidecar.normalization.target(ts).is_finite());
            } else {
                assert!(cell.is_none());
            }
        }
        // The Python preprocessor adds mutable embedding metadata after backfill.
        fs::write(pre.join("meta.json"), "{\"text_embeddings\": {}}\n").unwrap();
        assert!(Sidecar::open(&pre).is_ok());
        // Put the original valid legacy metadata back for subsequent backfill.
        fs::write(pre.join("meta.json"), legacy_meta).unwrap();
        let published = fs::read(pre.join(DATA)).unwrap();
        // A new non-null value where the archive had no datetime is rejected.
        parquet_dates(
            &raw.join("db/events.parquet"),
            &[source[0], source[1], source[2], Some(0)],
            true,
        );
        assert!(
            backfill(&raw, &pre, None)
                .unwrap_err()
                .to_string()
                .contains("content mismatch")
        );
        assert_eq!(fs::read(pre.join(DATA)).unwrap(), published);
        assert!(!pre.join(".datetime_exact.lock").exists());
        // A mutated immutable base index is rejected before sampling.
        let mut changed = fs::read(pre.join("column_index.json")).unwrap();
        changed.push(b' ');
        fs::write(pre.join("column_index.json"), changed).unwrap();
        assert!(
            Sidecar::open(&pre)
                .err()
                .unwrap()
                .to_string()
                .contains("stale datetime sidecar")
        );
        fs::write(pre.join("column_index.json"), &before[3]).unwrap();
        let mut corrupt = published.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        fs::write(pre.join(DATA), corrupt).unwrap();
        assert!(Sidecar::open(&pre).is_err());
    }
    #[test]
    fn constant_raw_dates_survive_legacy_nan_and_all_null_stays_missing() {
        for present in [true, false] {
            let temp = Temp::new();
            let raw = temp.0.join("raw");
            let output = temp.0.join("pre");
            fs::create_dir_all(raw.join("db")).unwrap();
            fs::write(
                raw.join("manifest.yaml"),
                "name: fixture\ntables:\n  events:\n    pkey: id\n    time_col: event\n",
            )
            .unwrap();
            let value = present.then_some(253_402_300_799_123_456);
            parquet_dates(&raw.join("db/events.parquet"), &[value, value], true);
            preprocess_exact(crate::pre::Cli {
                dataset_dir: raw.display().to_string(),
                out_dir: output.display().to_string(),
                source: None,
                skip_tasks: true,
                skip_db: false,
            })
            .unwrap();
            let pre = output.join("fixture");
            let meta: serde_json::Value =
                serde_json::from_reader(File::open(pre.join("meta.json")).unwrap()).unwrap();
            fs::write(
                pre.join("text_emb_fixture.bin"),
                vec![0; meta["num_text_strings"].as_u64().unwrap() as usize * 2],
            )
            .unwrap();
            backfill(&raw, &pre, None).unwrap();
            let sidecar = Sidecar::open(&pre).unwrap();
            assert_eq!(sidecar.normalization.std_seconds, 1.0);
            assert_eq!(sidecar.normalization.count, if present { 2 } else { 0 });
            let legacy = sampler(&pre, "events", false)
                .unwrap()
                .batch(Some(0), 0, 2, 4);
            assert!(legacy.batch_mask.iter().all(|v| !v));
            let exact = sampler(&pre, "events", true)
                .unwrap()
                .batch(Some(0), 0, 2, 4);
            assert!(exact.batch_mask.iter().all(|v| *v == present));
            let fields = exact.exact.as_ref().unwrap();
            assert_eq!(fields.valid.iter().filter(|v| **v).count() > 0, present);
            assert!(fields.target_values.iter().all(|v| *v == 0.0));
            for (i, valid) in fields.valid.iter().enumerate() {
                if *valid {
                    assert_eq!(fields.seconds[i], 253_402_300_799);
                    assert_eq!(fields.nanoseconds[i], 123_456_000);
                    assert!(exact.datetime_values[i].is_nan());
                }
            }
        }
    }
    fn task_dates(path: &Path, dates: &[i64]) {
        let mut df = DataFrame::new(vec![
            Series::new("event".into(), dates)
                .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                .unwrap()
                .into(),
            Series::new("entity".into(), vec![0_i64; dates.len()]).into(),
        ])
        .unwrap();
        ParquetWriter::new(File::create(path).unwrap())
            .finish(&mut df)
            .unwrap();
    }

    fn fixture(root: &Path, skip_tasks: bool, skip_db: bool, exact: bool) -> (PathBuf, PathBuf) {
        let raw = root.join("raw");
        let output = root.join("pre");
        fs::create_dir_all(raw.join("db")).unwrap();
        fs::create_dir_all(raw.join("tasks/forecast")).unwrap();
        fs::write(
            raw.join("manifest.yaml"),
            "name: fixture\ntables:\n  events:\n    pkey: id\n    time_col: event\n",
        )
        .unwrap();
        fs::write(
            raw.join("tasks/forecast/manifest.yaml"),
            "target_col: event\ntask_type: regression\nentity_col: entity\nentity_table: events\n",
        )
        .unwrap();
        parquet_dates(
            &raw.join("db/events.parquet"),
            &[Some(0), Some(1_000_000)],
            true,
        );
        task_dates(
            &raw.join("tasks/forecast/train.parquet"),
            &[2_000_000, 3_000_000],
        );
        let cli = crate::pre::Cli {
            dataset_dir: raw.display().to_string(),
            out_dir: output.display().to_string(),
            source: None,
            skip_tasks,
            skip_db,
        };
        if exact {
            preprocess_exact(cli).unwrap();
        } else {
            crate::pre::main(cli);
        }
        (raw, output.join("fixture"))
    }

    #[test]
    fn missing_receipt_and_explicit_historical_admission() {
        let temp = Temp::new();
        let (raw, pre) = fixture(&temp.0, false, false, false);
        assert!(!pre.join(SOURCE).exists());
        assert!(
            backfill(&raw, &pre, None)
                .unwrap_err()
                .to_string()
                .contains("missing source receipt")
        );
        assert!(!pre.join(META).exists());
        // Only a real new preprocessing invocation may issue a fresh receipt.
        preprocess_exact(crate::pre::Cli {
            dataset_dir: raw.display().to_string(),
            out_dir: pre.parent().unwrap().display().to_string(),
            source: None,
            skip_tasks: false,
            skip_db: false,
        })
        .unwrap();
        let mut receipt: SourceReceipt =
            serde_json::from_reader(File::open(pre.join(SOURCE)).unwrap()).unwrap();
        receipt.kind = "historical".into();
        receipt.origin = Origin {
            issuer: "fixture authority".into(),
            raw_revision: "fixture immutable raw snapshot".into(),
            archive_revision: "fixture immutable archive snapshot".into(),
            evidence: "test authority observed the prior preprocessing invocation".into(),
        };
        let external = temp.0.join("trusted.json");
        write_json(&external, &receipt).unwrap();
        backfill(&raw, &pre, Some(&external)).unwrap();
        assert_eq!(
            fs::read(&external).unwrap(),
            fs::read(pre.join(SOURCE)).unwrap()
        );
        assert!(
            backfill(&raw, &pre, None)
                .unwrap_err()
                .to_string()
                .contains("explicit")
        );
        validate(&pre).unwrap();
        assert!(Sidecar::open(&pre).is_ok());
        receipt.origin.raw_revision.clear();
        write_json(&external, &receipt).unwrap();
        assert!(
            backfill(&raw, &pre, Some(&external))
                .unwrap_err()
                .to_string()
                .contains("immutable revision")
        );
    }

    #[test]
    fn receipt_rejects_same_entity_swap_and_nonnull_edit_without_republishing() {
        let temp = Temp::new();
        let (raw, pre) = fixture(&temp.0, false, false, true);
        let before_meta = fs::read(pre.join(META)).unwrap();
        let before_payload = fs::read(pre.join(DATA)).unwrap();
        // FK anchors stay [0,0]; positional and non-null checks alone cannot catch this.
        for values in [[3_000_000, 2_000_000], [2_000_001, 3_000_000]] {
            task_dates(&raw.join("tasks/forecast/train.parquet"), &values);
            assert!(
                backfill(&raw, &pre, None)
                    .unwrap_err()
                    .to_string()
                    .contains("content mismatch")
            );
            assert_eq!(fs::read(pre.join(META)).unwrap(), before_meta);
            assert_eq!(fs::read(pre.join(DATA)).unwrap(), before_payload);
        }
        // The post-read check also rejects mutation after initial admission.
        let receipt: SourceReceipt =
            serde_json::from_reader(File::open(pre.join(SOURCE)).unwrap()).unwrap();
        let scratch = Scratch::new(&pre).unwrap();
        assert!(
            build_sidecar(
                &raw,
                &pre,
                &scratch,
                &receipt,
                &fs::read(pre.join(SOURCE)).unwrap(),
                base_identities(&pre).unwrap()
            )
            .unwrap_err()
            .to_string()
            .contains("content mismatch")
        );
        assert_eq!(fs::read(pre.join(META)).unwrap(), before_meta);
    }

    #[test]
    fn skip_policy_still_binds_complete_raw_inventory() {
        for (skip_tasks, skip_db) in [(true, false), (false, true)] {
            let temp = Temp::new();
            let (raw, pre) = fixture(&temp.0, skip_tasks, skip_db, true);
            validate(&pre).unwrap();
            let receipt: SourceReceipt =
                serde_json::from_reader(File::open(pre.join(SOURCE)).unwrap()).unwrap();
            assert_eq!(receipt.preprocess.skip_tasks, skip_tasks);
            assert_eq!(receipt.preprocess.skip_db, skip_db);
            assert!(
                receipt
                    .raw_files
                    .contains_key("tasks/forecast/train.parquet")
            );
            assert!(receipt.raw_files.contains_key("db/events.parquet"));
            let name = if skip_tasks {
                "tasks/forecast/train.parquet"
            } else {
                "db/events.parquet"
            };
            let bytes = fs::read(raw.join(name)).unwrap();
            fs::write(raw.join(name), b"changed ignored input").unwrap();
            assert!(
                backfill(&raw, &pre, None)
                    .unwrap_err()
                    .to_string()
                    .contains("content mismatch")
            );
            fs::write(raw.join(name), bytes).unwrap();
            let admitted = raw_snapshot(&raw).unwrap();
            fs::write(
                raw.join("tasks/forecast/test.parquet"),
                b"new ignored input",
            )
            .unwrap();
            assert!(
                stable_raw(&raw, &admitted)
                    .unwrap_err()
                    .to_string()
                    .contains("file manifest")
            );
            assert!(
                backfill(&raw, &pre, None)
                    .unwrap_err()
                    .to_string()
                    .contains("file manifest")
            );
        }
    }

    #[test]
    fn portable_validation_refreshes_only_stat_receipts_and_rejects_content_changes() {
        let temp = Temp::new();
        let (raw, pre) = fixture(&temp.0, false, false, true);
        fs::remove_dir_all(raw).unwrap(); // Validation must not need raw availability.
        let mut before = read_metadata(&pre).unwrap();
        let identities: Vec<_> = BASE_FILES
            .iter()
            .copied()
            .chain([DATA, SOURCE])
            .map(|name| (name, fs::read(pre.join(name)).unwrap()))
            .collect();
        for (name, _) in &identities {
            File::open(pre.join(name))
                .unwrap()
                .set_times(
                    std::fs::FileTimes::new()
                        .set_modified(UNIX_EPOCH + std::time::Duration::from_secs(1)),
                )
                .unwrap();
        }
        assert!(
            Sidecar::open(&pre)
                .err()
                .unwrap()
                .to_string()
                .contains("validate_datetime")
        );
        validate(&pre).unwrap();
        assert!(Sidecar::open(&pre).is_ok());
        for (name, bytes) in &identities {
            assert_eq!(&fs::read(pre.join(name)).unwrap(), bytes);
        }
        let after = read_metadata(&pre).unwrap();
        for (name, id) in &mut before.base_archive {
            id.modified_ns = after.base_archive[name].modified_ns;
        }
        before.payload.modified_ns = after.payload.modified_ns;
        before.source_receipt.modified_ns = after.source_receipt.modified_ns;
        assert_eq!(
            serde_json::to_value(before).unwrap(),
            serde_json::to_value(after).unwrap()
        );
        let meta = fs::read(pre.join(META)).unwrap();
        let mut old: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        old["format_version"] = 1.into();
        write_json(&pre.join(META), &old).unwrap();
        assert!(
            Sidecar::open(&pre)
                .err()
                .unwrap()
                .to_string()
                .contains("format_version=2")
        );
        assert!(
            validate(&pre)
                .unwrap_err()
                .to_string()
                .contains("format_version=2")
        );
        fs::write(pre.join(META), &meta).unwrap();
        let mut bytes = fs::read(pre.join(DATA)).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(pre.join(DATA), bytes).unwrap();
        assert!(
            validate(&pre)
                .unwrap_err()
                .to_string()
                .contains("content mismatch")
        );
        assert_eq!(fs::read(pre.join(META)).unwrap(), meta);
    }

    #[test]
    fn normalization_ieee_bits_survive_json_reader_and_repeated_validation() {
        // Exact source bits from ten observed fields in nine admitted archives.
        // Compare against these bits, not a second parse by the same JSON parser.
        let witnesses = [
            (0xc177_f14e_aad9_c217, 1.0_f64.to_bits()), // join-fec-campaign
            (0.0_f64.to_bits(), 0x421d_5831_62c2_9900), // join-spider-theme-gallery
            (0x418d_0061_07e7_ad1d, 1.0_f64.to_bits()), // plurel-3244
            (0.0_f64.to_bits(), 0x419e_6938_590c_7b0d), // plurel-3282
            (0x41ae_39b7_d014_f87b, 1.0_f64.to_bits()), // plurel-3664
            (0x413c_8e65_e2cc_c957, 1.0_f64.to_bits()), // plurel-3804
            (0x41aa_ff79_e02b_3808, 0x419f_4abb_cad0_965a), // plurel-4471, both fields
            (0.0_f64.to_bits(), 0x4199_0332_4646_22ae), // plurel-4874
            (0.0_f64.to_bits(), 0x41bd_4479_3740_ae67), // rel-f1
        ];
        fn without_mtimes(mut value: serde_json::Value) -> serde_json::Value {
            for id in value["base_archive"].as_object_mut().unwrap().values_mut() {
                id.as_object_mut().unwrap().remove("modified_ns");
            }
            for name in ["source_receipt", "payload"] {
                value[name].as_object_mut().unwrap().remove("modified_ns");
            }
            value
        }
        let temp = Temp::new();
        let (raw, pre) = fixture(&temp.0, false, false, true);
        fs::remove_dir_all(raw).unwrap();
        let protected: Vec<_> = BASE_FILES
            .iter()
            .copied()
            .chain([DATA, SOURCE])
            .map(|name| (name, fs::read(pre.join(name)).unwrap()))
            .collect();
        for (i, (mean_bits, std_bits)) in witnesses.into_iter().enumerate() {
            let mut meta = read_metadata(&pre).unwrap();
            // This tests wire preservation, not refitting this fixture's population.
            meta.normalization.mean_offset_seconds = f64::from_bits(mean_bits);
            meta.normalization.std_seconds = f64::from_bits(std_bits);
            let expected = without_mtimes(serde_json::to_value(&meta).unwrap());
            let encoded = serde_json::to_vec(&meta.normalization).unwrap();
            let parsed: Normalization = serde_json::from_slice(&encoded).unwrap();
            assert_eq!(parsed.mean_offset_seconds.to_bits(), mean_bits);
            assert_eq!(parsed.std_seconds.to_bits(), std_bits);
            write_json(&pre.join(META), &meta).unwrap();
            let opened = Sidecar::open(&pre).unwrap();
            assert_eq!(
                opened.normalization.mean_offset_seconds.to_bits(),
                mean_bits
            );
            assert_eq!(opened.normalization.std_seconds.to_bits(), std_bits);
            drop(opened);
            for pass in 0..3 {
                for (name, _) in &protected {
                    File::open(pre.join(name))
                        .unwrap()
                        .set_times(std::fs::FileTimes::new().set_modified(
                            UNIX_EPOCH
                                + std::time::Duration::from_secs(100 + (i * 3 + pass) as u64),
                        ))
                        .unwrap();
                }
                validate(&pre).unwrap();
                let after = read_metadata(&pre).unwrap();
                assert_eq!(after.normalization.mean_offset_seconds.to_bits(), mean_bits);
                assert_eq!(after.normalization.std_seconds.to_bits(), std_bits);
                assert_eq!(
                    without_mtimes(serde_json::to_value(&after).unwrap()),
                    expected
                );
                let opened = Sidecar::open(&pre).unwrap();
                assert_eq!(
                    opened.normalization.mean_offset_seconds.to_bits(),
                    mean_bits
                );
                assert_eq!(opened.normalization.std_seconds.to_bits(), std_bits);
                for (name, bytes) in &protected {
                    assert_eq!(&fs::read(pre.join(name)).unwrap(), bytes);
                }
            }
        }
    }

    #[test]
    fn full_validation_checks_interior_offsets_nanos_and_calendar_range() {
        let temp = Temp::new();
        let (_, pre) = fixture(&temp.0, false, false, true);
        let original_meta = fs::read(pre.join(META)).unwrap();
        let original_payload = fs::read(pre.join(DATA)).unwrap();
        let meta = read_metadata(&pre).unwrap();
        let (sec, ns, _) = layout(meta.num_nodes, meta.num_datetime_cells).unwrap();
        // Rebind the test metadata digest so these probe structure, not just the
        // earlier hash gate. This is not an authenticity claim for edited files.
        for (offset, value, expected) in [
            (HEADER + 8, 0_u64.to_le_bytes().to_vec(), "offset mismatch"),
            (ns, 1_000_000_000_u32.to_le_bytes().to_vec(), "nanoseconds"),
            (sec, i64::MAX.to_le_bytes().to_vec(), "calendar range"),
        ] {
            let mut payload = original_payload.clone();
            payload[offset..offset + value.len()].copy_from_slice(&value);
            fs::write(pre.join(DATA), payload).unwrap();
            let mut meta: Metadata = serde_json::from_slice(&original_meta).unwrap();
            meta.payload = identity(&pre.join(DATA)).unwrap();
            write_json(&pre.join(META), &meta).unwrap();
            let before = fs::read(pre.join(META)).unwrap();
            let error = validate(&pre).unwrap_err().to_string();
            assert!(error.contains(expected), "{error}");
            assert_eq!(fs::read(pre.join(META)).unwrap(), before);
        }
    }

    #[test]
    fn checked_archive_and_table_bounds_return_errors() {
        assert!(
            table_end(
                &TableInfo {
                    node_idx_offset: i32::MAX,
                    num_nodes: i32::MAX
                },
                2
            )
            .is_err()
        );
        assert!(
            table_end(
                &TableInfo {
                    node_idx_offset: -1,
                    num_nodes: 1
                },
                2
            )
            .is_err()
        );
        let bytes = rkyv::to_bytes::<Error>(&crate::common::Offsets { offsets: vec![0] }).unwrap();
        let offsets = rkyv::access::<ArchivedOffsets, Error>(&bytes).unwrap();
        assert!(archive_node(&[], offsets, 0).is_err());
        assert!(archive_node(&[], offsets, usize::MAX).is_err());
    }
}
