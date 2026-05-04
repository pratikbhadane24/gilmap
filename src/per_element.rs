//! Per-element Python C-API call loops. Caller holds an attached
//! `Python<'_>` token for the interpreter that owns `func_ptr`. Per
//! element: box → call → unbox → DECREF, with error fetch at each step.

use pyo3::ffi::*;
use pyo3::prelude::*;

/// # Safety
/// `func_ptr` must be a non-null `PyObject*` callable in the interpreter
/// owning `py`'s thread. `output.len() == input.len()`.
pub(crate) unsafe fn call_per_element_i64(
    py: Python<'_>,
    func_ptr: *mut PyObject,
    input: &[i64],
    output: &mut [i64],
) -> PyResult<()> {
    debug_assert_eq!(input.len(), output.len());
    unsafe {
        for j in 0..input.len() {
            let val = input[j];

            let val_obj = PyLong_FromLongLong(val as std::os::raw::c_longlong);
            if val_obj.is_null() {
                return Err(PyErr::fetch(py));
            }

            let res_obj = PyObject_CallOneArg(func_ptr, val_obj);
            Py_DECREF(val_obj);

            if res_obj.is_null() {
                return Err(PyErr::fetch(py));
            }

            let res_i64 = PyLong_AsLongLong(res_obj);
            Py_DECREF(res_obj);

            if res_i64 == -1 && !PyErr_Occurred().is_null() {
                return Err(PyErr::fetch(py));
            }

            output[j] = res_i64 as i64;
        }
    }
    Ok(())
}

/// # Safety
/// Same as [`call_per_element_i64`].
pub(crate) unsafe fn call_per_element_f64(
    py: Python<'_>,
    func_ptr: *mut PyObject,
    input: &[f64],
    output: &mut [f64],
) -> PyResult<()> {
    debug_assert_eq!(input.len(), output.len());
    unsafe {
        for j in 0..input.len() {
            let val = input[j];

            let val_obj = PyFloat_FromDouble(val as std::os::raw::c_double);
            if val_obj.is_null() {
                return Err(PyErr::fetch(py));
            }

            let res_obj = PyObject_CallOneArg(func_ptr, val_obj);
            Py_DECREF(val_obj);

            if res_obj.is_null() {
                return Err(PyErr::fetch(py));
            }

            let res_f64 = PyFloat_AsDouble(res_obj);
            Py_DECREF(res_obj);

            if res_f64 == -1.0 && !PyErr_Occurred().is_null() {
                return Err(PyErr::fetch(py));
            }

            output[j] = res_f64 as f64;
        }
    }
    Ok(())
}
