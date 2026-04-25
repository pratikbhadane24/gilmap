use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::ffi::*;
use arrow::array::{Array, Int64Array, make_array};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex, Condvar, OnceLock};

// Task sent to workers
struct Task {
    module_name: String,
    func_name: String,
    sys_path: Vec<String>,
    input_ptr: *const i64,
    output_ptr: *mut i64,
    len: usize,
    done: Arc<(Mutex<Option<Result<(), String>>>, Condvar)>,
}

unsafe impl Send for Task {}

enum WorkerMessage {
    Task(Task),
    Shutdown,
}

static WORKER_POOL: OnceLock<Vec<(Mutex<Sender<WorkerMessage>>, std::sync::Mutex<Option<std::thread::JoinHandle<()>>>)>> = OnceLock::new();

fn init_worker_pool() -> Vec<(Mutex<Sender<WorkerMessage>>, std::sync::Mutex<Option<std::thread::JoinHandle<()>>>)> {
    let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut pool = Vec::new();
    for _ in 0..num_threads {
        let (tx, rx) = channel::<WorkerMessage>();
        let handle = std::thread::spawn(move || {
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
                    eprintln!("Failed to create sub-interpreter");
                    return;
                }

                while let Ok(msg) = rx.recv() {
                    match msg {
                        WorkerMessage::Shutdown => break,
                        WorkerMessage::Task(task) => {
                            let thread_exec_result: Result<(), String> = Python::attach(|py_sub| {
                                let execute_inner = || -> PyResult<()> {
                                    let sys = py_sub.import("sys")?;
                                    let path = sys.getattr("path")?;
                                    
                                    // Add missing sys.paths for the task
                                    for p in &task.sys_path {
                                        let contains = path.call_method1("__contains__", (p.as_str(),))?.extract::<bool>()?;
                                        if !contains {
                                            path.call_method1("append", (p.as_str(),))?;
                                        }
                                    }

                                    let module = py_sub.import(task.module_name.as_str())?;
                                    let func = module.getattr(task.func_name.as_str())?;
                                    let func_ptr = func.as_ptr();

                                    let input_slice = std::slice::from_raw_parts(task.input_ptr, task.len);
                                    let output_slice = std::slice::from_raw_parts_mut(task.output_ptr, task.len);

                                    for j in 0..task.len {
                                        let val = input_slice[j];
                                        
                                        unsafe {
                                            let val_obj = PyLong_FromLongLong(val as std::os::raw::c_longlong);
                                            if val_obj.is_null() {
                                                return Err(PyErr::fetch(py_sub));
                                            }
                                            
                                            let res_obj = PyObject_CallOneArg(func_ptr, val_obj);
                                            Py_DECREF(val_obj);
                                            
                                            if res_obj.is_null() {
                                                return Err(PyErr::fetch(py_sub));
                                            }
                                            
                                            let res_i64 = PyLong_AsLongLong(res_obj);
                                            Py_DECREF(res_obj);
                                            
                                            if res_i64 == -1 && !PyErr_Occurred().is_null() {
                                                return Err(PyErr::fetch(py_sub));
                                            }
                                            
                                            output_slice[j] = res_i64 as i64;
                                        }
                                    }
                                    Ok(())
                                };
                                
                                match execute_inner() {
                                    Ok(_) => Ok(()),
                                    Err(e) => Err(format!("Python error in worker thread: {}", e)),
                                }
                            });

                            let (lock, cvar) = &*task.done;
                            let mut result_guard = lock.lock().unwrap();
                            *result_guard = Some(thread_exec_result);
                            cvar.notify_one();
                        }
                    }
                }

                Py_EndInterpreter(tstate);
            }
        });
        pool.push((Mutex::new(tx), std::sync::Mutex::new(Some(handle))));
    }
    pool
}

#[pyfunction]
fn execute<'py>(
    py: Python<'py>,
    module_name: &str,
    func_name: &str,
    array: Bound<'py, PyAny>,
    sys_path: Vec<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let pool = WORKER_POOL.get_or_init(init_worker_pool);
    let num_threads = pool.len();

    let array_data = arrow::array::ArrayData::from_pyarrow_bound(&array)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse Arrow array: {}", e)))?;
        
    let arrow_array = make_array(array_data);
    let len = arrow_array.len();
    
    let int_array = match arrow_array.as_any().downcast_ref::<Int64Array>() {
        Some(arr) => arr,
        None => return Err(PyRuntimeError::new_err("hyperfunctions currently only supports arrays of integers (Int64Array).")),
    };
    
    let chunk_size = if len == 0 { 1 } else { (len + num_threads - 1) / num_threads };
    let input_slice = int_array.values();
    
    let mut results = vec![0i64; len];
    
    let mut tasks_done = Vec::new();
    
    let chunks = input_slice.chunks(chunk_size);
    let mut_out_chunks = results.chunks_mut(chunk_size);
    
    let mut worker_idx = 0;
    
    for (chunk, mut_out_chunk) in chunks.zip(mut_out_chunks) {
        let done = Arc::new((Mutex::new(None), Condvar::new()));
        tasks_done.push(done.clone());
        
        let task = Task {
            module_name: module_name.to_string(),
            func_name: func_name.to_string(),
            sys_path: sys_path.clone(),
            input_ptr: chunk.as_ptr(),
            output_ptr: mut_out_chunk.as_mut_ptr(),
            len: chunk.len(),
            done,
        };
        
        // Distribute round-robin
        let (tx_mutex, _) = &pool[worker_idx % num_threads];
        let tx = tx_mutex.lock().unwrap();
        tx.send(WorkerMessage::Task(task)).map_err(|_| PyRuntimeError::new_err("Worker thread panicked or died"))?;
        worker_idx += 1;
    }
    
    // Release the GIL while waiting for workers to finish
    let wait_result = py.detach(|| {
        for done in tasks_done {
            let (lock, cvar) = &*done;
            let mut result_guard = lock.lock().unwrap();
            while result_guard.is_none() {
                result_guard = cvar.wait(result_guard).unwrap();
            }
            let res = result_guard.take().unwrap();
            if let Err(msg) = res {
                return Err(msg);
            }
        }
        Ok(())
    });
    
    match wait_result {
        Ok(_) => {
            let result_array = Int64Array::from(results);
            let result_data = result_array.into_data();
            result_data.to_pyarrow(py)
        },
        Err(msg) => Err(PyRuntimeError::new_err(msg)),
    }
}

#[pyfunction]
fn shutdown_workers(py: Python) {
    let _ = py.detach(|| {
        if let Some(pool) = WORKER_POOL.get() {
            // First send shutdown message to all workers
            for (mutex_tx, _) in pool {
                if let Ok(tx) = mutex_tx.lock() {
                    let _ = tx.send(WorkerMessage::Shutdown);
                }
            }
            
            // Then join them all to ensure Py_EndInterpreter is done safely
            for (_, mutex_handle) in pool {
                if let Ok(mut handle_opt) = mutex_handle.lock() {
                    if let Some(handle) = handle_opt.take() {
                        let _ = handle.join();
                    }
                }
            }
        }
    });
}

#[pymodule]
mod _hyperfunctions {
    use super::*;

    #[pymodule_export]
    use super::execute;

    #[pymodule_export]
    use super::shutdown_workers;
}
