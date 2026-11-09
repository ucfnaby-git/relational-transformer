use pyo3::prelude::*;

mod common;
pub mod fly;
mod pre;

#[pyfunction]
#[pyo3(signature = (dataset_dir, out_dir, *, source=None, skip_tasks=false, skip_db=false, exact_datetime=false))]
fn preprocess(
    py: Python<'_>,
    dataset_dir: String,
    out_dir: String,
    source: Option<String>,
    skip_tasks: bool,
    skip_db: bool,
    exact_datetime: bool,
) -> PyResult<()> {
    py.allow_threads(move || {
        let cli = pre::Cli {
            dataset_dir,
            out_dir,
            source,
            skip_tasks,
            skip_db,
        };
        if exact_datetime {
            fly::datetime::preprocess_exact(cli).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "preprocess(exact_datetime=True): {e}"
                ))
            })
        } else {
            pre::main(cli);
            Ok(())
        }
    })
}

/// Add exact timestamps to one existing dataset archive without rebuilding text embeddings.
#[pyfunction]
#[pyo3(signature = (raw_dir, pre_dir, *, source_receipt=None))]
fn backfill_datetime(
    py: Python<'_>,
    raw_dir: String,
    pre_dir: String,
    source_receipt: Option<String>,
) -> PyResult<()> {
    py.allow_threads(|| {
        fly::datetime::backfill(
            std::path::Path::new(&raw_dir),
            std::path::Path::new(&pre_dir),
            source_receipt.as_deref().map(std::path::Path::new),
        )
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "backfill_datetime(raw_dir={raw_dir:?}, pre_dir={pre_dir:?}): {e}"
            ))
        })
    })
}

/// Validate a copied sidecar completely and refresh only its local file states.
#[pyfunction]
fn validate_datetime(py: Python<'_>, pre_dir: String) -> PyResult<()> {
    py.allow_threads(|| {
        fly::datetime::validate(std::path::Path::new(&pre_dir)).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "validate_datetime(pre_dir={pre_dir:?}): {e}"
            ))
        })
    })
}

#[pymodule]
fn rustler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<fly::Sampler>()?;
    m.add_function(wrap_pyfunction!(preprocess, m)?)?;
    m.add_function(wrap_pyfunction!(backfill_datetime, m)?)?;
    m.add_function(wrap_pyfunction!(validate_datetime, m)?)?;

    Ok(())
}
