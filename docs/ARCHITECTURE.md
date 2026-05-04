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
                  └──┬──────┬──────┬──────┬──────┬─────────┘
                     │      │      │      │      │
                     ▼      ▼      ▼      ▼      ▼
                 ┌──────┐ ┌─────┐ ┌─────┐ ┌────┐ ┌──────────┐
                 │arrow_│ │numba│ │numba│ │jit │ │subinterp │
                 │kernel│ │cfunc│ │native│ │    │ │(fallback)│
                 └──┬───┘ └──┬──┘ └──┬──┘ └─┬──┘ └────┬─────┘
                    │        │       │      │         │
                    ▼        ▼       ▼      ▼         ▼
              ┌─────────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌─────────────────────┐
              │pyarrow. │ │raw   │ │numpy │ │Crane-│ │Rust worker pool     │
              │compute  │ │extern│ │round-│ │lift  │ │ GIL: sub-interps    │
              │batch    │ │"C"   │ │trip +│ │JIT   │ │ free-thread: rayon  │
              │chain    │ │fn ptr│ │numba │ │native│ │ + shared main interp│
              │         │ │+rayon│ │disp. │ │+rayon│ │ + adaptive chunk    │
              └─────────┘ └──────┘ └──────┘ └──────┘ └─────────────────────┘
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

### 3b. Worker-pool fallback (Rust)

`src/lib.rs` (~600 lines) owns two parallel executors that share contract
and chunking math, differing only in their parallelism primitive:

- **GIL-enabled CPython** → sub-interpreter pool (`execute`, the original
  PEP 684 design).
- **Free-threaded CPython** (`Py_GIL_DISABLED=1`) → rayon thread pool
  sharing the main interpreter (`execute_rayon`, `src/rayon_pool.rs`).
  Selected by `gilmap/__init__.py` based on `sysconfig.Py_GIL_DISABLED`.

The per-element FFI loop (`src/per_element.rs`) is shared by both paths,
so a behavior change applies uniformly.

Sub-interpreter pool architecture:

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

Free-threaded executor architecture (`src/rayon_pool.rs`):

- **No sub-interpreters.** PEP 684 own-GIL is incompatible with
  `Py_GIL_DISABLED`, so we share the main interpreter. Multiple worker
  threads can call the C-API concurrently — that's the no-GIL build's
  selling point.
- **Per-call resolve.** `module.func` is looked up once at the entry
  point; the resulting `Py<PyAny>` is shared (atomic refcount) across
  rayon tasks. No per-worker `func_cache` (it would be a global
  `Mutex<HashMap>` here, contended on every miss; the per-call import
  cost is dwarfed by the per-element FFI cost).
- **Parallelism via `rayon::scope`.** Adaptive chunking reuses the same
  `pick_chunk_size` heuristic as the sub-interp pool. Each chunk's
  closure calls `Python::attach` to establish a thread state, then runs
  the shared per-element loop.
- **GIL release.** The dispatcher wraps the rayon scope in
  `py.detach(...)` for symmetry with the sub-interp path; on the
  free-threaded build this is a no-op, on a hybrid GIL build it would
  serialize (the dispatcher is only routed here when free-threaded).

### 4. Lifecycle

- `gilmap.map` is the entire user surface.
- `_gilmap.shutdown_workers` is registered with `atexit` so the
  sub-interpreters tear down cleanly on process exit. The rayon pool is
  managed by rayon's global thread-pool and needs no explicit shutdown.

## File map

| File | Role |
|---|---|
| `gilmap/__init__.py` | public API + dispatch |
| `gilmap/_router.py` | backend detectors + decision cache |
| `gilmap/_jit.py` | Python-side AST → JSON IR walker for the JIT path |
| `src/lib.rs` | sub-interpreter worker pool, interned cache, adaptive chunking, JIT PyO3 bindings |
| `src/per_element.rs` | per-element Python C-API call loops shared by both worker pools |
| `src/rayon_pool.rs` | free-threaded executor: shared main interp + rayon |
| `src/cfunc_dispatch.rs` | numba `@cfunc` raw-pointer dispatcher (parallel `extern "C" fn(T)->T` over chunks) |
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

## Cranelift JIT (P5a + P5b + SIMD — shipped)

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

Supported IR:
- Expressions: `Constant`, `Name`, `BinOp` (incl. `%`, `//`, `**`),
  `UnaryOp`, `Compare`, `IfExpr`, `math.{sqrt,abs,fabs,exp,log,sin,cos,
  tan,floor,ceil}` calls.
- Statements: `Return`, `Assign`/`AugAssign` with per-local `i64`/`f64`
  inference, counted `for i in range(...)`, `while`, `if`/`else`
  (incl. early return), `break`, `continue`.

This closes `mandelbrot_iters` (gilmap is now the fastest runner across
all sizes — see `docs/BENCHMARKS.md`).

### SIMD path (P5c — shipped)

When `kernel.is_vectorizable_phase1()` returns true — single
`Stmt::Return` of a pure expression tree containing only
`Param`/`Const`/`BinOp(+,-,*)`/`Unary`/`Compare`/`IfExpr`/`Cast` —
`compile_simd` emits a dual loop:

1. **Vector main loop**: processes `SIMD_LANES` (= 2) elements per
   iteration. `F64X2` / `I64X2` loads, lane-wise arithmetic, vector
   store. Cranelift legalizes to native SSE2 (x86_64) or NEON (aarch64)
   instructions at codegen time.
2. **Scalar tail**: the trailing `len % LANES` elements run through the
   existing scalar `Lowerer` (no duplicated lowering logic).

Bodies that fail the predicate (locals, `for`/`while`, math calls,
integer mod/div, pow, mixed-statement bodies) fall through to the
existing scalar codegen — zero behavior change for those kernels.

Lane choice (Phase 1): 128-bit only. Universally legalized by Cranelift
across SSE2/AVX/AVX2/NEON. Wider 256-/512-bit lanes are a follow-up
gated on per-host ISA feature detection (see "Future moves").

## Future moves

- **Wider SIMD lanes** (P5c follow-up): Phase 1 uses 128-bit `F64X2` /
  `I64X2` for portability. AVX2 (256-bit) and AVX-512 (512-bit) hosts
  could double or quadruple throughput for pure-arithmetic kernels.
  Requires runtime ISA-feature detection and per-feature kernel cache
  keys.
