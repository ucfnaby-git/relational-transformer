//! Bounded context features preserving the finite FP32 bits already in archives.
use crate::common::{ArchivedNode, ArchivedSemType};
use half::bf16;
use itertools::izip;

pub const FEATURES: usize = 5;

pub fn source(node: &ArchivedNode, cell: usize, legacy_boolean: bool) -> Option<f32> {
    match node.sem_types[cell] {
        ArchivedSemType::Number => Some(node.number_values[cell].into()),
        ArchivedSemType::Boolean if !legacy_boolean => Some(node.boolean_values[cell].into()),
        _ => None,
    }
}

pub fn features(value: f32) -> Option<[bf16; FEATURES]> {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    if exponent == 0x7f80_0000 {
        return None;
    }
    // Avoid DAZ/FTZ-dependent arithmetic on signed zero or source subnormals.
    let cue = if exponent == 0 {
        bf16::ZERO
    } else {
        let x = f64::from(value);
        bf16::from_f64(x / (1.0 + x.abs()))
    };
    Some([
        cue,
        bf16::from_f32(((bits >> 24) & 255) as f32 / 256.0),
        bf16::from_f32(((bits >> 16) & 255) as f32 / 256.0),
        bf16::from_f32(((bits >> 8) & 255) as f32 / 256.0),
        bf16::from_f32((bits & 255) as f32 / 256.0),
    ])
}

pub struct Vecs {
    pub features: Vec<bf16>,
    pub target_values: Vec<f32>,
}

pub struct Slices<'a> {
    pub features: &'a mut [bf16],
    pub target_values: &'a mut [f32],
}

impl Vecs {
    pub fn new(len: usize) -> Self {
        Self {
            features: vec![bf16::ZERO; len * FEATURES],
            target_values: vec![0.0; len],
        }
    }

    pub fn chunks_exact_mut(&mut self, len: usize) -> impl Iterator<Item = Slices<'_>> {
        izip!(
            self.features.chunks_exact_mut(len * FEATURES),
            self.target_values.chunks_exact_mut(len)
        )
        .map(|(features, target_values)| Slices {
            features,
            target_values,
        })
    }
}

impl Slices<'_> {
    pub fn clear(&mut self) {
        self.features.fill(bf16::ZERO);
        self.target_values.fill(0.0);
    }
}

#[cfg(test)]
#[path = "numeric_tests.rs"]
mod tests;
