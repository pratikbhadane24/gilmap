# gilmap

`gilmap` is a Python + Rust parallel map runtime for numeric, single-argument Python functions.

It combines:

- An auto-router that picks the fastest backend per callable (numba native, pyarrow.compute kernels, Cranelift JIT, or sub-interpreter pool)
- Rust worker threads + Python sub-interpreters (PEP 684, one per worker) for the general fallback
- Apache Arrow arrays for efficient Python <-> Rust transfer

Public API: `gilmap.map(func, iterable, *, debug=False)` and `gilmap.explain(func)`.

See `docs/ARCHITECTURE.md` for the layered design and `docs/BACKENDS.md` for the per-backend contract.

## What it is optimized for

`gilmap.map` is best for **CPU-bound** functions where each element does enough work to amortize scheduling/conversion overhead.

It is usually a poor fit for:

- tiny per-element work (overhead dominates)
- I/O-bound callables
- lambdas/local functions/not-importable functions

## Install

```bash
pip install gilmap
```

Prebuilt wheels are published for CPython 3.12 and 3.13 on Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (x64). No Rust toolchain required.

> **Note:** gilmap requires a standard GIL-enabled CPython build. Free-threaded builds (`cp313t`) are not supported because per-worker sub-interpreters with their own GIL are incompatible with PEP 703.

## Requirements

- CPython `>= 3.12` (GIL-enabled build)
- `pyarrow >= 14.0.0` (installed automatically)

## Quick start

### Integer input

```python
# tasks.py
def square(x: int) -> int:
    return x * x
```

```python
# app.py
import gilmap
from tasks import square

out = gilmap.map(square, [1, 2, 3, 4])
print(out)  # [1, 4, 9, 16]
```

### Float input

```python
def affine(x: float) -> float:
    return x * 1.5 + 0.5

print(gilmap.map(affine, [1.0, 2.0, 3.0]))
```

### Arrow input (Arrow output)

```python
import pyarrow as pa
import gilmap
from tasks import square

arr = pa.array([1, 2, 3, 4], type=pa.int64())
out = gilmap.map(square, arr)
print(type(out))  # pyarrow.Array
```

## API reference

### `gilmap.map(func, iterable, *, debug=False) -> list | pyarrow.Array`

Executes `func` over `iterable` in parallel while preserving element order.

#### Parameters

- `func`: callable accepting one numeric argument and returning one numeric value
- `iterable`: either
  - a Python iterable of values castable to `int64` or `float64`, or
  - a `pyarrow.Array`
- `debug`: if `True` (or env `GILMAP_DEBUG=1`), prints the chosen backend per call

#### Return behavior

- If input is a Python iterable: returns a Python `list`
- If input is a `pyarrow.Array`: returns a `pyarrow.Array`

#### Type behavior

- Float input (`float32`/`float64`) is cast to `float64`
- Non-float input is cast to `int64`
- Backend owns the output dtype. Most backends preserve the input lane; JIT may return `int64` for an `f64` input when the body does `return int(...)`.

### `gilmap.explain(func) -> dict`

Returns the router's decision without executing — `{"backend", "reason", "has_fast_path"}`. Useful to verify which path will fire.

```python
gilmap.explain(lambda x: x * 2.0 + 1.0)
# {'backend': 'arrow_kernel', 'reason': '...', 'has_fast_path': True}
```

## Callable constraints

Constraints depend on the backend the router picks:

- **`numba_native` / `arrow_kernel` / `jit` fast paths**: lambdas, `__main__` functions, and locals are all fine — the body is lowered to native or pyarrow.compute and never imported into a worker.
- **`subinterp` fallback**: workers import by module + function name inside sub-interpreters, so lambdas, `<locals>`, and `__main__` functions are rejected. Move them to an importable module.

Use `gilmap.explain(func)` to see which backend will run.

## Error model

| Exception | Condition |
| --- | --- |
| `TypeError` | `func` is not callable |
| `ValueError` | lambda/local/`__main__` function routed to the `subinterp` fallback |
| `TypeError` | input cannot be cast to supported numeric Arrow type |
| `RuntimeError` | execution/import failure in worker sub-interpreters; or `subinterp` fallback hit on a free-threaded (`Py_GIL_DISABLED`) build |

If any worker fails, the whole call fails and no partial result is returned.

## Architecture

`gilmap.map` runs an auto-router (`gilmap/_router.py`) that picks one of four backends per callable; the decision is cached on a weakref so repeat calls are O(1).

| Backend | Selected when | Where work runs |
|---|---|---|
| `numba_native` | `func` exposes numba dispatcher metadata or a cfunc `address` | numpy round-trip + cached numba dispatcher |
| `arrow_kernel` | body is a single `return <expr>` of supported binops/unops/`math.*` | one or more `pyarrow.compute` C calls — whole array at once |
| `jit` | same shape as `arrow_kernel` plus `%`/`//`/ternary, where arrow_kernel rejects | Cranelift-compiled `extern "C" fn(*const T, *mut T, len)`, hash-cached |
| `subinterp` | nothing else matches | Rust worker pool, one PEP 684 sub-interpreter per thread, interned `(module, name)` cache, adaptive chunk scheduling |

Full per-backend contract is in `docs/BACKENDS.md`; the layered execution flow (Python entry → router → fast-path / Rust pool, plus the JIT pipeline) is in `docs/ARCHITECTURE.md`.

## Worker lifecycle and shutdown

- Sub-interpreter worker threads are long-lived after first use and reused across calls
- `shutdown_workers` sends one shutdown message per worker and joins threads
- `shutdown_workers` is automatically registered with `atexit`
- JIT-compiled kernels are cached by IR hash for the lifetime of the process

## Testing

```bash
# Rust checks
cargo clippy --all-targets --all-features
cargo test

# Python tests (after maturin develop)
python -m pytest -q
```

## Benchmarking

Numbers below are regenerated from a real harness run — see `docs/BENCHMARKS.md`
for the full report (per-workload tables, charts, and a mandatory "where gilmap
loses" section). The block between the markers is overwritten by
`python -m benchmarks.report <results.json>`; do not hand-edit it.

<!-- BENCH:START -->
On `quick_collatz` at N=1,000,000, **gilmap** (gilmap_arrow@arrow) is **170.22× faster** than the best non-gilmap runner (`numba`) — measured on Apple M3 Max.

**Summary across 6 workloads:**

| workload | best gilmap variant | speedup vs runner-up | runner-up |
|---|---|---|---|
| quick_collatz N=1,000,000 | gilmap_arrow@arrow | 170.22× | numba |
| float_math N=1,000,000 | gilmap_arrow@arrow | 85.72× | numba |
| float_math N=100,000 | gilmap_arrow@arrow | 65.80× | numba |
| quick_collatz N=100,000 | gilmap_arrow@arrow | 46.96× | numba |
| count_primes N=128 | gilmap_arrow@arrow | 43.21× | mp_pool |

⚠ rows are workloads where gilmap loses to a faster runner — included so this table is honest, not cherry-picked. Full breakdown in `docs/BENCHMARKS.md`.

<!-- BENCH:END -->

### What the suite measures

`benchmarks/` runs a sweep across input size, lane (int64/float64), and container
(list/Arrow) and compares `gilmap.map` against:

- `map` (single-thread baseline)
- `multiprocessing.Pool` and `concurrent.futures.ProcessPoolExecutor`
- `concurrent.futures.ThreadPoolExecutor` (GIL-bound; included to show why
  naive threading fails for CPU work)
- `joblib.Parallel` (loky), `ray`, `dask` — when installed
- `numpy` vector form and `numba @njit(parallel=True)` for vectorizable
  workloads — included so the report can honestly show where gilmap loses

Each cell is warmed up once and timed N=3 times by default (`--repeats`
flag to override); results are byte-equivalent to the `map` baseline
(1e-6 float tolerance) or the cell fails. First-call warmup cost and
per-runner setup cost are reported separately, not amortized into
steady-state numbers.

### Reproduce

```bash
pip install -e ".[bench]"
python -m benchmarks.run --out benchmarks/results/$(hostname)-$(date +%Y%m%d).json
python -m benchmarks.report benchmarks/results/<file>.json
```

The report generator refuses to publish if no losses are recorded — the
"Where gilmap loses" section is mandatory so the marketing stays honest.

## Known limitations

- Single-argument callables only
- Numeric lanes only (`int64` / `float64`)
- Callable must be importable by module + name (no lambda/local/`__main__`)
- Null handling in input arrays is not currently modeled as nullable output semantics

## Repository layout

```text
.
├── Cargo.toml
├── pyproject.toml
├── gilmap/
│   ├── __init__.py         # public API: map(), explain()
│   ├── _router.py          # backend detectors + weakref decision cache
│   ├── _jit.py             # Python AST → typed JSON IR for the JIT path
│   └── _ast_utils.py       # shared AST helpers
├── src/
│   ├── lib.rs              # sub-interpreter worker pool + PyO3 bindings
│   ├── ast_ir.rs           # typed IR shared with _jit.py
│   └── jit.rs              # Cranelift codegen + per-kernel registry
├── tests/
│   ├── tasks.py
│   ├── test_router.py
│   ├── test_jit.py
│   ├── test_numba_bridge.py
│   ├── test_free_threaded.py
│   ├── test_parallel.py
│   ├── test_safety.py
│   └── test_heavy.py
├── benchmarks/             # dev-only benchmark suite
│   ├── workloads.py
│   ├── runners.py
│   ├── harness.py
│   ├── suite.py
│   ├── run.py
│   └── report.py
└── docs/
    ├── ARCHITECTURE.md     # layered design, JIT pipeline
    ├── BACKENDS.md         # per-backend contract
    └── BENCHMARKS.md       # generated; full benchmark report
```

## CI and packaging

The GitHub Actions workflow builds wheels/sdists for multiple platforms using maturin.

## Build from source

Only needed for development on gilmap itself, or to install on a platform without a prebuilt wheel.

Prerequisites:

- Rust toolchain (`cargo`, `rustc`)
- `maturin`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip maturin pytest
maturin develop
```

`maturin develop` builds and installs the `_gilmap` extension for the active environment.
