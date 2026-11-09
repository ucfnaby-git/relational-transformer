use super::*;
use crate::common::{Adj, Node, Offsets, SemType, TableInfo};
use crate::fly::Sampler;
use rkyv::rancor::Error;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const PATTERNS: &[u32] = &[
    0,
    0x8000_0000,
    1,
    0x8000_0001,
    0x007f_ffff,
    0x807f_ffff,
    0x0080_0000,
    0x8080_0000,
    0x0080_0001,
    0x8080_0001,
    0x3f80_0000,
    0xbf80_0000,
    0x3f80_00ff,
    0x3f80_0100,
    0x3f7f_ffff,
    0x3f80_0001,
    0x7f7f_ffff,
    0xff7f_ffff,
];

fn decode(features: &[bf16]) -> u32 {
    features[1..].iter().fold(0, |bits, digit| {
        (bits << 8) | (digit.to_f32() * 256.0) as u32
    })
}

fn check_bits(bits: u32) {
    let features = features(f32::from_bits(bits)).unwrap();
    assert_eq!(decode(&features), bits);
    for feature in features {
        assert!(feature.is_finite() && feature.to_f32().abs() <= 1.0);
        assert!(feature.to_bits() & 0x7fff == 0 || feature.to_bits() & 0x7f80 != 0);
    }
    if bits & 0x7f80_0000 == 0 {
        assert_eq!(features[0].to_bits(), 0);
    }
}

#[test]
fn bounded_codec_preserves_finite_source_bits() {
    for &bits in PATTERNS {
        check_bits(bits);
    }
    let mut bits = 20261109_u32;
    for _ in 0..100_000 {
        bits = bits.wrapping_mul(1664525).wrapping_add(1013904223);
        if bits & 0x7f80_0000 != 0x7f80_0000 {
            check_bits(bits);
        }
    }
    for bits in [0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0x7f80_0001] {
        assert!(features(f32::from_bits(bits)).is_none());
    }
}

struct Fixture(PathBuf);
impl Fixture {
    fn new(patterns: &[u32]) -> Self {
        let root = std::env::temp_dir().join(format!(
            "rt-numeric-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let pre = root.join("fixture");
        fs::create_dir_all(&pre).unwrap();
        let mut nodes = Vec::new();
        let mut tables = HashMap::new();
        for table in ["numbers", "booleans", "hidden"] {
            let offset = nodes.len();
            let values = if table == "hidden" {
                &patterns[..1]
            } else {
                patterns
            };
            for &bits in values {
                let value = f32::from_bits(bits);
                nodes.push(Node {
                    node_idx: nodes.len() as i32,
                    table_name_idx: 0,
                    col_name_idxs: vec![1, 2],
                    sem_types: vec![
                        match table {
                            "numbers" => SemType::Number,
                            "booleans" => SemType::Boolean,
                            _ => SemType::Text,
                        },
                        SemType::Text,
                    ],
                    // Irrelevant slots deliberately contain infinity; boolean routing must not read these.
                    number_values: vec![
                        if table == "numbers" {
                            value
                        } else {
                            f32::INFINITY
                        },
                        f32::NEG_INFINITY,
                    ],
                    boolean_values: vec![
                        if table == "booleans" {
                            value
                        } else {
                            f32::INFINITY
                        },
                        f32::INFINITY,
                    ],
                    datetime_values: vec![0.0, 0.0],
                    text_values: vec![0, 0],
                    class_value_idx: vec![-1, -1],
                    ..Default::default()
                });
            }
            tables.insert(
                format!("{table}:Db"),
                TableInfo {
                    node_idx_offset: offset as i32,
                    num_nodes: values.len() as i32,
                },
            );
        }
        let context_offset = nodes.len();
        let context_value = if f32::from_bits(patterns[0]).is_finite() {
            f32::INFINITY
        } else {
            f32::from_bits(patterns[0])
        };
        nodes.push(Node {
            node_idx: context_offset as i32,
            table_name_idx: 0,
            col_name_idxs: vec![1, 2],
            sem_types: vec![SemType::Number, SemType::Number],
            number_values: vec![0.5, context_value],
            boolean_values: vec![f32::INFINITY; 2],
            datetime_values: vec![0.0; 2],
            text_values: vec![0; 2],
            class_value_idx: vec![-1; 2],
            ..Default::default()
        });
        tables.insert(
            "context:Db".into(),
            TableInfo {
                node_idx_offset: context_offset as i32,
                num_nodes: 1,
            },
        );
        let mut bytes = Vec::new();
        let mut offsets = vec![0_i64];
        for node in &nodes {
            bytes.extend_from_slice(&rkyv::to_bytes::<Error>(node).unwrap());
            offsets.push(bytes.len() as i64);
        }
        fs::write(pre.join("nodes.rkyv"), bytes).unwrap();
        fs::write(
            pre.join("offsets.rkyv"),
            rkyv::to_bytes::<Error>(&Offsets { offsets }).unwrap(),
        )
        .unwrap();
        fs::write(
            pre.join("p2f_adj.rkyv"),
            rkyv::to_bytes::<Error>(&Adj {
                adj: vec![Vec::new(); nodes.len()],
            })
            .unwrap(),
        )
        .unwrap();
        fs::write(
            pre.join("table_info.json"),
            serde_json::to_vec(&tables).unwrap(),
        )
        .unwrap();
        fs::write(pre.join("text.json"), b"[\"fixture\",\"value\",\"hidden\"]").unwrap();
        fs::write(pre.join("column_index.json"), b"{\"value\":1,\"hidden\":2}").unwrap();
        fs::write(pre.join("text_emb_fixture.bin"), [0_u8; 6]).unwrap();
        fs::write(pre.join("meta.json"), serde_json::to_vec(&serde_json::json!({"format_version":1,"name":"fixture","num_nodes":nodes.len(),"num_text_strings":3})).unwrap()).unwrap();
        fs::write(pre.join("expected.json"), serde_json::to_vec(&serde_json::json!({"patterns":patterns,"number_rows":patterns.len(),"boolean_offset":patterns.len(),"hidden_offset":patterns.len()*2,"columns":2})).unwrap()).unwrap();
        Self(root)
    }

    fn sampler(&self, table: &str, exact: bool, legacy_boolean: bool, ignore: bool) -> Sampler {
        let tables: HashMap<String, TableInfo> =
            serde_json::from_slice(&fs::read(self.0.join("fixture/table_info.json")).unwrap())
                .unwrap();
        let info = &tables[&format!("{table}:Db")];
        Sampler::new_impl(
            vec![(
                "fixture".into(),
                table.into(),
                info.node_idx_offset,
                info.num_nodes,
            )],
            0,
            0,
            1,
            vec![3],
            vec![2],
            0,
            0,
            vec![false],
            0.0,
            "fixture",
            self.0.display().to_string(),
            1,
            0,
            0,
            vec![1],
            vec![vec![]],
            vec![None],
            -1,
            true,
            ignore,
            0,
            false,
            legacy_boolean,
            5.0,
            None,
            false,
            exact,
        )
        .unwrap()
    }

    fn export(&self, destination: &Path) {
        fs::create_dir_all(destination).unwrap();
        let pre = destination.join("fixture");
        fs::create_dir(&pre).unwrap();
        for file in fs::read_dir(self.0.join("fixture")).unwrap() {
            let file = file.unwrap();
            fs::copy(file.path(), pre.join(file.file_name())).unwrap();
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn native_source_routing_hidden_slots_and_legacy_parity() {
    let fixture = Fixture::new(PATTERNS);
    for table in ["numbers", "booleans", "hidden"] {
        for legacy_boolean in [false, true] {
            let exact = fixture
                .sampler(table, true, legacy_boolean, false)
                .batch(Some(0), 0, PATTERNS.len(), 4)
                .unwrap();
            let legacy = fixture
                .sampler(table, false, legacy_boolean, false)
                .batch(Some(0), 0, PATTERNS.len(), 4)
                .unwrap();
            assert!(legacy.numeric.is_none());
            assert_eq!(exact.node_idxs, legacy.node_idxs);
            assert_eq!(exact.sem_types, legacy.sem_types);
            assert_eq!(exact.is_targets, legacy.is_targets);
            assert_eq!(exact.is_padding, legacy.is_padding);
            assert_eq!(exact.batch_mask, legacy.batch_mask);
            assert_eq!(
                exact
                    .number_values
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                legacy
                    .number_values
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
            let values = exact.numeric.as_ref().unwrap();
            for slot in 0..exact.node_idxs.len() {
                let f = &values.features[slot * FEATURES..(slot + 1) * FEATURES];
                let expected =
                    if !exact.is_padding[slot] && exact.sem_types[slot] == SemType::Number as i32 {
                        let row = exact.node_idxs[slot] as usize % PATTERNS.len();
                        Some(PATTERNS[row])
                    } else {
                        None
                    };
                if let Some(bits) = expected {
                    assert_eq!(decode(f), bits);
                    assert_eq!(values.target_values[slot].to_bits(), bits);
                } else {
                    assert!(f.iter().all(|x| x.to_bits() == 0));
                    assert_eq!(values.target_values[slot].to_bits(), 0);
                }
            }
        }
    }
    if let Ok(path) = std::env::var("RT_NUMERIC_TRANSPORT_FIXTURE") {
        fixture.export(Path::new(&path));
    }
}

#[test]
fn infinity_is_fatal_and_nan_stays_missing() {
    for bits in [0x7f80_0000, 0xff80_0000] {
        let fixture = Fixture::new(&[bits]);
        for table in ["numbers", "booleans", "context"] {
            for ignore in [false, true] {
                let sampler = fixture.sampler(table, true, false, ignore);
                for batch_idx in [Some(0), None] {
                    let result = sampler.batch(batch_idx, 0, 1, 4);
                    let error = result.err().expect("infinity must return, not drop/retry");
                    assert!(
                        error.contains("infinite numeric source")
                            && error.contains("dataset \"fixture\"")
                            && error.contains(&format!("bits=0x{bits:08x}"))
                    );
                }
            }
        }
        if let Ok(path) = std::env::var("RT_NUMERIC_ERROR_FIXTURES") {
            fixture.export(&Path::new(&path).join(if bits == 0x7f80_0000 {
                "positive_infinity"
            } else {
                "negative_infinity"
            }));
        }
        // Legacy mode remains able to emit its original infinite BF16 value.
        assert!(
            fixture
                .sampler("numbers", false, false, false)
                .batch(Some(0), 0, 1, 4)
                .is_ok()
        );
    }
    let fixture = Fixture::new(&[0x7fc0_0000]);
    if let Ok(path) = std::env::var("RT_NUMERIC_ERROR_FIXTURES") {
        fixture.export(&Path::new(&path).join("nan"));
    }
    let context = fixture
        .sampler("context", true, false, false)
        .batch(Some(0), 0, 1, 4)
        .unwrap();
    assert_eq!(context.batch_mask, [true]);
    assert_eq!(
        context.numeric.as_ref().unwrap().target_values[0].to_bits(),
        0.5_f32.to_bits()
    );
    assert!(context.is_padding[1]);
    assert!(
        context.numeric.as_ref().unwrap().features[FEATURES..]
            .iter()
            .all(|x| x.to_bits() == 0)
    );
    let batch = fixture
        .sampler("numbers", true, false, false)
        .batch(Some(0), 0, 1, 4)
        .unwrap();
    assert_eq!(batch.batch_mask, [false]);
    assert!(batch.is_padding.iter().all(|x| *x));
    let values = batch.numeric.unwrap();
    assert!(values.features.iter().all(|x| x.to_bits() == 0));
    assert!(values.target_values.iter().all(|x| x.to_bits() == 0));
}

#[cfg(target_arch = "x86_64")]
#[test]
#[allow(deprecated)]
fn native_codec_and_targets_on_actual_rayon_threads_with_ftz_daz() {
    use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
    let fixture = Fixture::new(PATTERNS);
    for enabled in [false, true] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.broadcast(|_| {
            struct Restore(u32);
            impl Drop for Restore {
                fn drop(&mut self) {
                    unsafe {
                        _mm_setcsr(self.0);
                    }
                }
            }
            let saved = unsafe { _mm_getcsr() };
            let _restore = Restore(saved);
            let wanted = if enabled {
                saved | 0x8040
            } else {
                saved & !0x8040
            };
            unsafe {
                _mm_setcsr(wanted);
            }
            assert_eq!(
                unsafe { _mm_getcsr() } & 0x8040,
                if enabled { 0x8040 } else { 0 }
            );
            for &bits in PATTERNS {
                check_bits(bits);
            }
            let sampler = fixture.sampler("numbers", true, false, false);
            let batch = sampler.batch(Some(0), 0, PATTERNS.len(), 4).unwrap();
            let values = batch.numeric.unwrap();
            for slot in 0..batch.node_idxs.len() {
                if batch.sem_types[slot] == SemType::Number as i32 && !batch.is_padding[slot] {
                    let bits = PATTERNS[batch.node_idxs[slot] as usize];
                    assert_eq!(values.target_values[slot].to_bits(), bits);
                    assert_eq!(
                        decode(&values.features[slot * FEATURES..(slot + 1) * FEATURES]),
                        bits
                    );
                }
            }
        });
    }
}
