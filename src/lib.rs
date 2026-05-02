use arrow::array::{Array, Float64Array, Int64Array, make_array};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use crossbeam_channel::{Sender, unbounded};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi::*;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

type TaskResult = Result<(), String>;
type TaskDone = Arc<(Mutex<Option<TaskResult>>, Condvar)>;
type WorkerHandle = std::sync::Mutex<Option<std::thread::JoinHandle<()>>>;
type WorkerPool = (Sender<WorkerMessage>, Vec<WorkerHandle>);

#[derive(Clone, Copy)]
enum DataType {
    Int64 {
        input_ptr: *const i64,
        output_ptr: *mut i64,
    },
    Float64 {
        input_ptr: *const f64,
        output_ptr: *mut f64,
    },
}
unsafe impl Send for DataType {}

// Task sent to workers
struct Task {
    module_name: String,
    func_name: String,
    sys_path: Vec<String>,
    data: DataType,
    len: usize,
    done: TaskDone,
}

unsafe impl Send for Task {}

enum WorkerMessage {
    Task(Task),
    Shutdown,
}

static WORKER_POOL: OnceLock<WorkerPool> = OnceLock::new();

fn wait_for_tasks(tasks_done: Vec<TaskDone>) -> TaskResult {
    let mut first_error: Option<String> = None;

    for done in tasks_done {
        let (lock, cvar) = &*done;
        let mut result_guard = lock.lock().unwrap();
        while result_guard.is_none() {
            result_guard = cvar.wait(result_guard).unwrap();
        }

        if let Err(err) = result_guard.take().unwrap()
            && first_error.is_none()
        {
            first_error = Some(err);
        }
    }

    match first_error {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

fn init_worker_pool() -> WorkerPool {
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let (tx, rx) = unbounded::<WorkerMessage>();
    let mut pool = Vec::new();

    for _ in 0..num_threads {
        let rx_clone = rx.clone();
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

                let mut func_cache: HashMap<(String, String), Py<PyAny>> = HashMap::new();
                let mut path_set: HashSet<String> = HashSet::new();

                loop {
                    let msg = rx_clone.recv();
                    match msg {
                        Ok(WorkerMessage::Shutdown) | Err(_) => {
                            func_cache.clear(); // Drop any cached PyObjects before Py_EndInterpreter
                            break;
                        }
                        Ok(WorkerMessage::Task(task)) => {
                            let thread_exec_result: Result<(), String> = Python::attach(|py_sub| {
                                let mut execute_inner = || -> PyResult<()> {
                                    let func_obj = if let Some(func) = func_cache
                                        .get(&(task.module_name.clone(), task.func_name.clone()))
                                    {
                                        func.clone_ref(py_sub)
                                    } else {
                                        let sys = py_sub.import("sys")?;
                                        let path = sys.getattr("path")?;

                                        // Add missing sys.paths for the task efficiently (O(1) checks locally)
                                        for p in &task.sys_path {
                                            if !path_set.contains(p) {
                                                let p_str = p.as_str();
                                                let contains = path
                                                    .call_method1("__contains__", (p_str,))?
                                                    .extract::<bool>()?;
                                                if !contains {
                                                    path.call_method1("append", (p_str,))?;
                                                }
                                                path_set.insert(p.clone());
                                            }
                                        }

                                        let module = py_sub.import(task.module_name.as_str())?;
                                        let func = module.getattr(task.func_name.as_str())?;
                                        let func_obj = func.unbind();
                                        func_cache.insert(
                                            (task.module_name.clone(), task.func_name.clone()),
                                            func_obj.clone_ref(py_sub),
                                        );
                                        func_obj
                                    };

                                    let func_ptr = func_obj.as_ptr();

                                    match task.data {
                                        DataType::Int64 {
                                            input_ptr,
                                            output_ptr,
                                        } => {
                                            let input_slice =
                                                std::slice::from_raw_parts(input_ptr, task.len);
                                            let output_slice = std::slice::from_raw_parts_mut(
                                                output_ptr, task.len,
                                            );

                                            for j in 0..task.len {
                                                let val = input_slice[j];

                                                let val_obj = PyLong_FromLongLong(
                                                    val as std::os::raw::c_longlong,
                                                );
                                                if val_obj.is_null() {
                                                    return Err(PyErr::fetch(py_sub));
                                                }

                                                let res_obj =
                                                    PyObject_CallOneArg(func_ptr, val_obj);
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
                                        DataType::Float64 {
                                            input_ptr,
                                            output_ptr,
                                        } => {
                                            let input_slice =
                                                std::slice::from_raw_parts(input_ptr, task.len);
                                            let output_slice = std::slice::from_raw_parts_mut(
                                                output_ptr, task.len,
                                            );

                                            for j in 0..task.len {
                                                let val = input_slice[j];

                                                let val_obj = PyFloat_FromDouble(
                                                    val as std::os::raw::c_double,
                                                );
                                                if val_obj.is_null() {
                                                    return Err(PyErr::fetch(py_sub));
                                                }

                                                let res_obj =
                                                    PyObject_CallOneArg(func_ptr, val_obj);
                                                Py_DECREF(val_obj);

                                                if res_obj.is_null() {
                                                    return Err(PyErr::fetch(py_sub));
                                                }

                                                let res_f64 = PyFloat_AsDouble(res_obj);
                                                Py_DECREF(res_obj);

                                                if res_f64 == -1.0 && !PyErr_Occurred().is_null() {
                                                    return Err(PyErr::fetch(py_sub));
                                                }

                                                output_slice[j] = res_f64 as f64;
                                            }
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
        pool.push(std::sync::Mutex::new(Some(handle)));
    }
    (tx, pool)
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
    let num_threads = pool.1.len();

    let array_data = arrow::array::ArrayData::from_pyarrow_bound(&array)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse Arrow array: {}", e)))?;

    let arrow_array = make_array(array_data);
    let len = arrow_array.len();

    let is_float = arrow_array.data_type() == &arrow::datatypes::DataType::Float64;

    let chunk_size = if len == 0 {
        1
    } else {
        len.div_ceil(num_threads)
    };
    let mut tasks_done = Vec::new();

    if is_float {
        let float_array = arrow_array.as_any().downcast_ref::<Float64Array>().unwrap();
        let input_slice = float_array.values();
        let mut results = vec![0f64; len];

        let chunks = input_slice.chunks(chunk_size);
        let mut_out_chunks = results.chunks_mut(chunk_size);

        for (chunk, mut_out_chunk) in chunks.zip(mut_out_chunks) {
            let done = Arc::new((Mutex::new(None), Condvar::new()));
            tasks_done.push(done.clone());

            let task = Task {
                module_name: module_name.to_string(),
                func_name: func_name.to_string(),
                sys_path: sys_path.clone(),
                data: DataType::Float64 {
                    input_ptr: chunk.as_ptr(),
                    output_ptr: mut_out_chunk.as_mut_ptr(),
                },
                len: chunk.len(),
                done,
            };

            // Push work to the shared queue
            let tx = &pool.0;
            tx.send(WorkerMessage::Task(task))
                .map_err(|_| PyRuntimeError::new_err("Worker thread panicked or died"))?;
        }

        let wait_result = py.detach(|| wait_for_tasks(tasks_done));

        match wait_result {
            Ok(_) => {
                let result_array = Float64Array::from(results);
                let result_data = result_array.into_data();
                result_data.to_pyarrow(py)
            }
            Err(msg) => Err(PyRuntimeError::new_err(msg)),
        }
    } else {
        let int_array = match arrow_array.as_any().downcast_ref::<Int64Array>() {
            Some(arr) => arr,
            None => {
                return Err(PyRuntimeError::new_err(
                    "gilmap currently only supports arrays of int64 or float64.",
                ));
            }
        };

        let input_slice = int_array.values();
        let mut results = vec![0i64; len];

        let chunks = input_slice.chunks(chunk_size);
        let mut_out_chunks = results.chunks_mut(chunk_size);

        for (chunk, mut_out_chunk) in chunks.zip(mut_out_chunks) {
            let done = Arc::new((Mutex::new(None), Condvar::new()));
            tasks_done.push(done.clone());

            let task = Task {
                module_name: module_name.to_string(),
                func_name: func_name.to_string(),
                sys_path: sys_path.clone(),
                data: DataType::Int64 {
                    input_ptr: chunk.as_ptr(),
                    output_ptr: mut_out_chunk.as_mut_ptr(),
                },
                len: chunk.len(),
                done,
            };

            // Push work to the shared queue
            let tx = &pool.0;
            tx.send(WorkerMessage::Task(task))
                .map_err(|_| PyRuntimeError::new_err("Worker thread panicked or died"))?;
        }

        let wait_result = py.detach(|| wait_for_tasks(tasks_done));

        match wait_result {
            Ok(_) => {
                let result_array = Int64Array::from(results);
                let result_data = result_array.into_data();
                result_data.to_pyarrow(py)
            }
            Err(msg) => Err(PyRuntimeError::new_err(msg)),
        }
    }
}

#[pyfunction]
fn shutdown_workers(py: Python) {
    py.detach(|| {
        if let Some((tx, pool)) = WORKER_POOL.get() {
            let num_threads = pool.len();
            // Send exactly one shutdown message per worker thread
            for _ in 0..num_threads {
                let _ = tx.send(WorkerMessage::Shutdown);
            }

            // Then join them all to ensure Py_EndInterpreter is done safely
            for mutex_handle in pool {
                if let Ok(mut handle_opt) = mutex_handle.lock()
                    && let Some(handle) = handle_opt.take()
                {
                    let _ = handle.join();
                }
            }
        }
    });
}

#[pymodule]
mod _gilmap {
    #[pymodule_export]
    use super::execute;

    #[pymodule_export]
    use super::shutdown_workers;
}
