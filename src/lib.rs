use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::ffi::*;
use arrow::array::{Array, Int64Array, make_array};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};

#[pyfunction]
fn execute<'py>(
    py: Python<'py>,
    module_name: &str,
    func_name: &str,
    array: Bound<'py, PyAny>,
    sys_path: Vec<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let array_data = arrow::array::ArrayData::from_pyarrow_bound(&array)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse Arrow array: {}", e)))?;
        
    let arrow_array = make_array(array_data);
    let len = arrow_array.len();
    
    let int_array = match arrow_array.as_any().downcast_ref::<Int64Array>() {
        Some(arr) => arr,
        None => return Err(PyRuntimeError::new_err("hyperfunctions currently only supports arrays of integers (Int64Array).")),
    };
    
    let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let chunk_size = if len == 0 { 1 } else { (len + num_threads - 1) / num_threads };
    
    let input_slice = int_array.values();
    let module_name = module_name.to_string();
    let func_name = func_name.to_string();
    
    let thread_result = py.detach(|| {
        let mut results = vec![0i64; len];
        
        let exec_result = std::thread::scope(|s| -> Result<(), String> {
            let mut handles = vec![];
            
            let chunks = input_slice.chunks(chunk_size);
            let mut_out_chunks = results.chunks_mut(chunk_size);
            
            for (chunk, mut_out_chunk) in chunks.zip(mut_out_chunks) {
                let mod_name = module_name.clone();
                let fn_name = func_name.clone();
                let paths = sys_path.clone();
                
                handles.push(s.spawn(move || -> Result<(), String> {
                    unsafe {
                        let config = PyInterpreterConfig {
                            use_main_obmalloc: 0,
                            allow_fork: 0,
                            allow_exec: 0,
                            allow_threads: 1,
                            allow_daemon_threads: 0,
                            check_multi_interp_extensions: 1,
                            gil: PyInterpreterConfig_OWN_GIL,
                        };
                        let mut tstate: *mut PyThreadState = std::ptr::null_mut();
                        let status = Py_NewInterpreterFromConfig(&mut tstate, &config);
                        if PyStatus_Exception(status) != 0 {
                            return Err("Failed to create sub-interpreter".to_string());
                        }
                        
                        let thread_exec_result: Result<(), String> = Python::attach(|py_sub| {
                            let execute_inner = || -> PyResult<()> {
                                let sys = py_sub.import("sys")?;
                                let path = sys.getattr("path")?;
                                for p in paths {
                                    path.call_method1("append", (p,))?;
                                }

                                let module = py_sub.import(&mod_name)?;
                                let func = module.getattr(&fn_name)?;
                                
                                for (j, &val) in chunk.iter().enumerate() {
                                    let res = func.call1((val,))?;
                                    let res_i64: i64 = res.extract()?;
                                    mut_out_chunk[j] = res_i64;
                                }
                                Ok(())
                            };
                            
                            match execute_inner() {
                                Ok(_) => Ok(()),
                                Err(e) => Err(format!("Python error in worker thread: {}", e)),
                            }
                        });
                        
                        Py_EndInterpreter(tstate);
                        
                        thread_exec_result
                    }
                }));
            }
            
            for handle in handles {
                handle.join().map_err(|_| "Rust thread panicked".to_string())??;
            }
            
            Ok(())
        });
        
        match exec_result {
            Ok(_) => Ok(results),
            Err(e) => Err(e),
        }
    });
    
    match thread_result {
        Ok(results) => {
            let result_array = Int64Array::from(results);
            let result_data = result_array.into_data();
            result_data.to_pyarrow(py)
        },
        Err(msg) => Err(PyRuntimeError::new_err(msg)),
    }
}

#[pymodule]
mod _hyperfunctions {
    use super::*;

    #[pymodule_export]
    use super::execute;
}
