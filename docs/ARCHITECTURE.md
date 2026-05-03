# gilmap architecture

```
                        ┌─────────────────────────────┐
                        │   gilmap.map(func, iter)    │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │  gilmap/_router.py :: decide(func)     │
                  │  one-shot AST + attribute introspection │
                  │  decision cached on a weakref(func)    │
                  └────┬────────────┬─────────────┬────────┘
                       │            │             │
                       ▼            ▼             ▼
               ┌────────────┐ ┌────────────┐ ┌────────────┐
               │ numba_     │ │ arrow_     │ │ subinterp  │
               │ native     │ │ kernel     │ │ (fallback) │
               └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
                     │              │              │
                     ▼              ▼              ▼
            ┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
            │ numpy round-    │ │ pyarrow.     │ │ Rust worker pool │
            │ trip + cached   │ │ compute.X    │ │ (PEP 684)        │
            │ numba dispatch  │ │ batch chain  │ │ + adaptive chunk │
            └─────────────────┘ └──────────────┘ │ + work-steal Q   │
                                                  │ + interned cache │
                                                  └──────────────────┘
```

## Layers

### 1. Entry: `gilmap.map`

`gilmap/__init__.py` does three things and gets out of the way:
1. Validates `func` is callable.
2. Calls `_router.decide(func)`.
3. Either invokes the chosen backend's `fast_path` or falls back to the
   sub-interpreter pool via the Rust extension `_gilmap.execute`.

Input is normalized to a `pyarrow.Array` (`int64` or `float64`); the cast
is skipped when dtype already matches (P1 Tier-1).

### 2. Router

`gilmap/_router.py` contains:
- `BackendDecision` dataclass: `(backend, reason, fast_path)`.
- A weakref-keyed cache so repeat calls hit O(1).
- Detectors run in priority order; first match wins.

See `docs/BACKENDS.md` for the per-backend contract.

### 3a. Fast paths (Python-side)

- `numba_native`: `func` carries numba metadata. We call it via numpy
  round-trip; numba's specialization cache is hit per element.
- `arrow_kernel`: a single-`return` body of supported binops/unops/math
  calls is compiled at decide-time into a closure of `pyarrow.compute`
  kernels. Whole array → one C call (or a short chain of them).

Both paths produce a `pyarrow.Array` and skip the Rust worker pool.

### 3b. Sub-interpreter fallback (Rust)

`src/lib.rs` (≈400 lines) owns the worker pool. Architecture:

- **One pool per process**, lazy-initialized via `OnceLock` on first call.
- **One sub-interpreter per worker thread** via
  `Py_NewInterpreterFromConfig` with `PyInterpreterConfig_OWN_GIL`
  (PEP 684).
- **Work distribution**: bounded crossbeam channel (cap = 2 ×
  num_threads × chunks_per_worker). Adaptive chunk sizing
  (`pick_chunk_size`) so we always have ≥ num_threads × 4 chunks for the
  shared queue → fast workers naturally steal from busy peers.
- **Per-worker state** (P1 Tier-1):
  - `func_cache: HashMap<(Arc<str>, Arc<str>), Py<PyAny>>` — interned
    name keys; pointer-equal lookups in the hot path.
  - `last_func` short-circuit cache: same `Arc<CallContext>` →
    pointer-eq check skips even the hash lookup.
  - `last_sys_path_id: u64` → `sys.path` is reimported only when the
    caller's path-list hash changes.
- **Per-call shared metadata** is folded into one `Arc<CallContext>`;
  every `Task` carries a single Arc clone instead of three String
  allocations (P1 Tier-1).
- **Element loop**: one `PyObject_CallOneArg` per element.
  `PyLong_FromLongLong` / `PyFloat_FromDouble` → call → DECREF the arg.
  4 FFI calls per element. Future move D (Cranelift JIT) replaces this
  loop wholesale for whitelisted ASTs.
- **Result**: `Float64Array::from(results)` / `Int64Array::from(results)`
  → `to_pyarrow(py)` once at the end.

### 4. Lifecycle

- `gilmap.map` is the entire user surface.
- `_gilmap.shutdown_workers` is registered with `atexit` so the
  sub-interpreters tear down cleanly on process exit.

## File map

| File | Role |
|---|---|
| `gilmap/__init__.py` | public API + dispatch |
| `gilmap/_router.py` | backend detectors + decision cache |
| `gilmap/_jit.py` | Python-side AST → JSON IR walker for the JIT path |
| `src/lib.rs` | sub-interpreter worker pool, interned cache, adaptive chunking, JIT PyO3 bindings |
| `src/ast_ir.rs` | typed IR shared between `_jit.py` and Cranelift codegen |
| `src/jit.rs` | Cranelift JIT module + per-kernel registry |
| `tests/test_router.py` | router selection + fast-path correctness |
| `tests/test_jit.py` | JIT correctness vs Python parity |
| `tests/test_numba_bridge.py` | `numba_native` dispatch round-trip |
| `tests/test_free_threaded.py` | `Py_GIL_DISABLED` gating |
| `tests/test_safety.py` | sub-interp guard rails |
| `tests/test_parallel.py` | end-to-end parallelism check |
| `benchmarks/` | comparator suite + report generator |
| `docs/BACKENDS.md` | per-backend contract |
| `docs/BENCHMARKS.md` | generated benchmark report |

## Cranelift JIT (P5a — shipped)

`src/jit.rs` + `src/ast_ir.rs` host the Cranelift codegen. Pipeline:

```
gilmap/_jit.py    Python AST  →  walked into typed dict  →  json.dumps
                                                                    │
                                                                    ▼
src/lib.rs        jit_compile(json)  →  serde_json::from_str  →  Kernel
                                                                    │
                                                                    ▼
src/jit.rs        compile()  →  Cranelift IR with inner i in 0..len loop
                  → JITModule.finalize  →  raw fn pointer cached by hash
                                                                    │
                                                                    ▼
src/lib.rs        jit_apply(hash, dtype, arr)  →  invoke_kernel
                  →  unsafe transmute to extern "C" fn(*const T, *mut T, usize)
                  →  return Float64Array / Int64Array
```

v1 covers single `return <expr>`. P5b extends the IR (`Assign`, `For`,
`IfStmt`, `Locals`) and the Cranelift codegen to handle multi-statement
bodies — the missing piece for `mandelbrot_iters` and friends.

## Future moves

- **P5b — JIT for multi-statement bodies**: extend `Expr` enum with
  `Block`, `Assign`, `For(range)`, `IfElse(early_return)`. Codegen
  builds basic blocks for control flow. Closes the `mandelbrot_iters`
  loss (currently 13× behind numba).
- **Move B** (free-threaded primary): rayon thread pool sharing the main
  interpreter when `Py_GIL_DISABLED=1`. Today the router still picks the
  fast paths on free-threaded builds; only the sub-interp fallback raises.
- **Move C extension** (numba native pointer): pull `func.address` from
  `@cfunc` and call the raw `extern "C"` pointer from Rust to skip the
  numpy round-trip entirely.
- **SIMD lanes inside JIT loops**: today the JITed body processes one
  scalar per iteration. Vectorize with `std::simd<f64; 8>` /
  `std::simd<i64; 4>` for measurable wins on the f64 lane.
