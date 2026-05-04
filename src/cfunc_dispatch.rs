//! Raw-pointer dispatch for numba `@cfunc` callables.
//!
//! When the router identifies a `@cfunc(types.<scalar>(types.<scalar>))`
//! decorator, it pulls `func.address` (a stable `extern "C"` function pointer
//! the numba runtime owns for the process lifetime) and routes through
//! `cfunc_apply` (in `src/lib.rs`) → here. Bypasses the numpy round-trip
//! that the legacy `numba_native` path uses.
//!
//! Parallelism mirrors `jit::invoke_kernel_parallel` — same chunk math
//! (`pick_chunk_size`), same rayon::scope. The kernel ABI is scalar-in /
//! scalar-out (`extern "C" fn(T) -> T`), so the per-element loop lives in
//! this module rather than dispatching to a single buffer-shaped kernel.

use crate::pick_chunk_size;

/// `extern "C"` ABI for `@cfunc(types.float64(types.float64))`.
type CFuncF64 = unsafe extern "C" fn(f64) -> f64;

/// `extern "C"` ABI for `@cfunc(types.int64(types.int64))`.
type CFuncI64 = unsafe extern "C" fn(i64) -> i64;

/// Apply a numba cfunc pointer over an f64 input slice in parallel.
///
/// # Safety
/// `addr` must be the address of an `extern "C" fn(f64) -> f64` valid for
/// the duration of the call. numba `CFunc.address` provides this for the
/// process lifetime.
pub(crate) unsafe fn apply_f64(addr: usize, input: &[f64], output: &mut [f64]) {
    let f: CFuncF64 = unsafe { std::mem::transmute(addr) };
    apply_in_parallel(input, output, |xs, ys| {
        for (i, &x) in xs.iter().enumerate() {
            ys[i] = unsafe { f(x) };
        }
    });
}

/// Apply a numba cfunc pointer over an i64 input slice in parallel.
///
/// # Safety
/// Same as [`apply_f64`] for the i64 ABI.
pub(crate) unsafe fn apply_i64(addr: usize, input: &[i64], output: &mut [i64]) {
    let f: CFuncI64 = unsafe { std::mem::transmute(addr) };
    apply_in_parallel(input, output, |xs, ys| {
        for (i, &x) in xs.iter().enumerate() {
            ys[i] = unsafe { f(x) };
        }
    });
}

/// Generic chunk-parallel driver: split `input`/`output` into rayon-spawned
/// chunks of `pick_chunk_size` elements and run `body(in_chunk, out_chunk)`
/// per chunk. Single-threaded fast path when the input is shorter than the
/// rayon thread count (spawn overhead would dominate).
fn apply_in_parallel<T, F>(input: &[T], output: &mut [T], body: F)
where
    T: Send + Sync,
    F: Fn(&[T], &mut [T]) + Send + Sync,
{
    debug_assert_eq!(input.len(), output.len());
    let len = input.len();
    if len == 0 {
        return;
    }
    let num_threads = rayon::current_num_threads().max(1);
    if len < num_threads {
        body(input, output);
        return;
    }
    let chunk_size = pick_chunk_size(len, num_threads);
    rayon::scope(|s| {
        for (in_chunk, out_chunk) in input.chunks(chunk_size).zip(output.chunks_mut(chunk_size))
        {
            let body = &body;
            s.spawn(move |_| body(in_chunk, out_chunk));
        }
    });
}
