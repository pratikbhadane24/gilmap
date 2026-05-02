use arrow::array::{Array, Float64Array, Int64Array, make_array};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use crossbeam_channel::{Sender, bounded};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi::*;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

type TaskResult = Result<(), String>;
type TaskDone = Arc<(Mutex<Option<TaskResult>>, Condvar)>;
type WorkerHandle = std::sync::Mutex<Option<std::thread::JoinHandle<()>>>;
type WorkerPool = (Sender<WorkerMessage>, Vec<WorkerHandle>);

// Per-call shared metadata (module name, func name, sys.path) lifted out of
// `Task` so that each chunk only carries a single Arc clone instead of
// re-allocating the strings/Vec on every dispatch. Hot path on small inputs
// where many chunks share identical metadata.
struct CallContext {
    module_name: Arc<str>,
    func_name: Arc<str>,
    sys_path: Arc<Vec<String>>,
    // Identity hash of sys_path; workers compare this against the previous
    // call's hash to skip the per-task `sys.path` import + scan when callers
    // haven't changed it.
    sys_path_id: u64,
}

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

struct Task {
    ctx: Arc<CallContext>,
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

// Tier-2: aim for many chunks (so fast workers steal from busy workers via
// the shared channel) without making chunks so small that per-chunk dispatch
// overhead eats the gains.
//
// Formula: num_chunks = min(len, max(num_threads * 4, ceil(len / MAX_CHUNK))).
// Then chunk_size = ceil(len / num_chunks).
//
// - For len < num_threads*4 (e.g. test_parallel.py's len=10), chunks = len
//   so each worker still gets one element — preserves small-input parallelism.
// - For mid-size, chunks = num_threads*4 — ample stealing surface.
// - For huge len, chunks scale up so chunk_size stays under MAX_CHUNK.
const CHUNKS_PER_WORKER: usize = 4;
const MAX_CHUNK: usize = 4096;

fn pick_chunk_size(len: usize, num_threads: usize) -> usize {
    if len == 0 {
        return 1;
    }
    let want = (num_threads * CHUNKS_PER_WORKER).max(len.div_ceil(MAX_CHUNK));
    let num_chunks = len.min(want.max(1));
    len.div_ceil(num_chunks)
}

// Cheap-and-stable identity hash for sys.path so workers can detect a
// no-change call and skip the `import sys; sys.path` round-trip entirely.
fn sys_path_id(paths: &[String]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    paths.len().hash(&mut h);
    for p in paths {
        p.hash(&mut h);
    }
    h.finish()
}

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
    // Tier-2: bounded channel sized to ~2× per-worker chunk depth so we
    // get back-pressure (no megabytes of queued tasks for huge N) while
    // still allowing workers to pipeline ahead of the dispatcher.
    let queue_cap = num_threads * CHUNKS_PER_WORKER * 2;
    let (tx, rx) = bounded::<WorkerMessage>(queue_cap);
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

                // Tier-1: cache by Arc<str> identity. Same (module, func)
                // strings interned at the dispatcher share Arc storage, so
                // hash + Eq compare pointer-equal Arcs in O(1) without
                // allocating String keys per lookup.
                let mut func_cache: HashMap<(Arc<str>, Arc<str>), Py<PyAny>> = HashMap::new();
                let mut path_set: HashSet<String> = HashSet::new();
                // Tier-1: track most recent (ctx_ptr, func) so back-to-back
                // chunks of the same call hit a single pointer compare instead
                // of even a hash lookup.
                let mut last_ctx_ptr: *const CallContext = std::ptr::null();
                let mut last_func: Option<Py<PyAny>> = None;
                // Tier-1: skip sys.path scan when caller hasn't changed it.
                let mut last_sys_path_id: u64 = u64::MAX;

                loop {
                    let msg = rx_clone.recv();
                    match msg {
                        Ok(WorkerMessage::Shutdown) | Err(_) => {
                            func_cache.clear();
                            // last_func dropped by going out of scope; explicit
                            // None-assign would be dead code (clippy).
                            break;
                        }
                        Ok(WorkerMessage::Task(task)) => {
                            let thread_exec_result: Result<(), String> = Python::attach(|py_sub| {
                                let ctx = task.ctx.clone();
                                let mut execute_inner = || -> PyResult<()> {
                                    let ctx_ptr = Arc::as_ptr(&ctx);
                                    let func_obj: Py<PyAny> = if ctx_ptr == last_ctx_ptr
                                        && let Some(f) = &last_func
                                    {
                                        f.clone_ref(py_sub)
                                    } else if let Some(f) = func_cache
                                        .get(&(ctx.module_name.clone(), ctx.func_name.clone()))
                                    {
                                        let cloned = f.clone_ref(py_sub);
                                        last_ctx_ptr = ctx_ptr;
                                        last_func = Some(cloned.clone_ref(py_sub));
                                        cloned
                                    } else {
                                        // sys.path append only if call brought new entries
                                        if ctx.sys_path_id != last_sys_path_id {
                                            let sys = py_sub.import("sys")?;
                                            let path = sys.getattr("path")?;
                                            for p in ctx.sys_path.iter() {
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
                                            last_sys_path_id = ctx.sys_path_id;
                                        }

                                        let module = py_sub.import(ctx.module_name.as_ref())?;
                                        let func = module.getattr(ctx.func_name.as_ref())?;
                                        let func_obj = func.unbind();
                                        func_cache.insert(
                                            (ctx.module_name.clone(), ctx.func_name.clone()),
                                            func_obj.clone_ref(py_sub),
                                        );
                                        last_ctx_ptr = ctx_ptr;
                                        last_func = Some(func_obj.clone_ref(py_sub));
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

// Process-wide intern table for module+function names so repeated calls with
// the same callable share Arc<str> storage (no per-call allocation, and the
// worker pointer-equality fast path actually fires).
static NAME_INTERN: OnceLock<Mutex<HashMap<String, Arc<str>>>> = OnceLock::new();

fn intern(name: &str) -> Arc<str> {
    let table = NAME_INTERN.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = table.lock().unwrap();
    if let Some(s) = guard.get(name) {
        return s.clone();
    }
    let arc: Arc<str> = Arc::from(name);
    guard.insert(name.to_string(), arc.clone());
    arc
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

    let chunk_size = pick_chunk_size(len, num_threads);

    // Build the per-call CallContext once and Arc-clone into each Task.
    let ctx = Arc::new(CallContext {
        module_name: intern(module_name),
        func_name: intern(func_name),
        sys_path_id: sys_path_id(&sys_path),
        sys_path: Arc::new(sys_path),
    });

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
                ctx: ctx.clone(),
                data: DataType::Float64 {
                    input_ptr: chunk.as_ptr(),
                    output_ptr: mut_out_chunk.as_mut_ptr(),
                },
                len: chunk.len(),
                done,
            };

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
                ctx: ctx.clone(),
                data: DataType::Int64 {
                    input_ptr: chunk.as_ptr(),
                    output_ptr: mut_out_chunk.as_mut_ptr(),
                },
                len: chunk.len(),
                done,
            };

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
            for _ in 0..num_threads {
                let _ = tx.send(WorkerMessage::Shutdown);
            }

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
