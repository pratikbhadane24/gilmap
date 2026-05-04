//! Free-threaded executor: shares the main interpreter, parallelizes via
//! rayon. Used in place of the sub-interpreter pool when CPython is built
//! with `Py_GIL_DISABLED=1` (PEP 703) — `Py_NewInterpreterFromConfig` with
//! own-GIL is incompatible with the no-GIL build.

use arrow::array::{Array, Float64Array, Int64Array, make_array};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi::PyObject;
use pyo3::prelude::*;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use crate::per_element::{call_per_element_f64, call_per_element_i64};
use crate::pick_chunk_size;

/// Process-wide cache of the last `sys.path` hash patched into the main
/// interpreter. Lets `execute_rayon` skip the per-call import + scan when
/// callers haven't mutated `sys.path` between invocations.
static LAST_SYS_PATH_ID: Mutex<u64> = Mutex::new(u64::MAX);

fn sys_path_id(paths: &[String]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    paths.len().hash(&mut h);
    for p in paths {
        p.hash(&mut h);
    }
    h.finish()
}

#[pyfunction]
pub(crate) fn execute_rayon<'py>(
    py: Python<'py>,
    module_name: &str,
    func_name: &str,
    array: Bound<'py, PyAny>,
    sys_path: Vec<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let path_id = sys_path_id(&sys_path);
    {
        let mut last = LAST_SYS_PATH_ID.lock().unwrap();
        if *last != path_id {
            let sys = py.import("sys")?;
            let path = sys.getattr("path")?;
            for p in &sys_path {
                let contains: bool = path
                    .call_method1("__contains__", (p.as_str(),))?
                    .extract()?;
                if !contains {
                    path.call_method1("append", (p.as_str(),))?;
                }
            }
            *last = path_id;
        }
    }

    let module = py.import(module_name)?;
    let func: Py<PyAny> = module.getattr(func_name)?.unbind();

    let array_data = arrow::array::ArrayData::from_pyarrow_bound(&array)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse Arrow array: {}", e)))?;
    let arrow_array = make_array(array_data);
    let len = arrow_array.len();
    let is_float = arrow_array.data_type() == &arrow::datatypes::DataType::Float64;

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let chunk_size = pick_chunk_size(len, num_threads);

    if is_float {
        let float_array = arrow_array
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let input_slice = float_array.values();
        let mut results = vec![0f64; len];

        let exec_result = py.detach(|| {
            run_chunks(input_slice, &mut results, chunk_size, &func, call_per_element_f64)
        });
        match exec_result {
            Ok(()) => Float64Array::from(results).into_data().to_pyarrow(py),
            Err(msg) => Err(PyRuntimeError::new_err(msg)),
        }
    } else {
        let int_array = arrow_array
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| {
                PyRuntimeError::new_err(
                    "gilmap currently only supports arrays of int64 or float64.",
                )
            })?;
        let input_slice = int_array.values();
        let mut results = vec![0i64; len];

        let exec_result = py.detach(|| {
            run_chunks(input_slice, &mut results, chunk_size, &func, call_per_element_i64)
        });
        match exec_result {
            Ok(()) => Int64Array::from(results).into_data().to_pyarrow(py),
            Err(msg) => Err(PyRuntimeError::new_err(msg)),
        }
    }
}

/// Generic chunk-parallel driver: spawn rayon tasks of `chunk_size`
/// elements, each running `call_elem(py, func_ptr, in_chunk, out_chunk)`
/// on a freshly-attached Python token. Returns the first worker error
/// observed (if any).
fn run_chunks<T>(
    input: &[T],
    output: &mut [T],
    chunk_size: usize,
    func: &Py<PyAny>,
    call_elem: unsafe fn(Python<'_>, *mut PyObject, &[T], &mut [T]) -> PyResult<()>,
) -> Result<(), String>
where
    T: Send + Sync,
{
    if input.is_empty() {
        return Ok(());
    }
    let first_err: Mutex<Option<String>> = Mutex::new(None);

    rayon::scope(|s| {
        for (in_chunk, out_chunk) in input
            .chunks(chunk_size)
            .zip(output.chunks_mut(chunk_size))
        {
            let first_err_ref = &first_err;
            s.spawn(move |_| {
                let result: PyResult<()> = Python::attach(|py| unsafe {
                    call_elem(py, func.as_ptr(), in_chunk, out_chunk)
                });
                if let Err(e) = result {
                    let mut g = first_err_ref.lock().unwrap();
                    if g.is_none() {
                        *g = Some(format!("Python error in rayon worker: {}", e));
                    }
                }
            });
        }
    });

    match first_err.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
