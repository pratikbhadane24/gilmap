# gilmap

`gilmap` is a Python + Rust parallel map runtime for numeric, single-argument, module-level Python functions.

It combines:

- Rust worker threads for CPU parallelism
- Python sub-interpreters (one per worker thread)
- Apache Arrow arrays for efficient Python <-> Rust transfer

The public API is one function: `gilmap.map`.

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

### `gilmap.map(func, iterable) -> list | pyarrow.Array`

Executes `func` over `iterable` in parallel while preserving element order.

#### Parameters

- `func`: callable accepting one numeric argument and returning one numeric value
- `iterable`: either
  - a Python iterable of values castable to `int64` or `float64`, or
  - a `pyarrow.Array`

#### Return behavior

- If input is a Python iterable: returns a Python `list`
- If input is a `pyarrow.Array`: returns a `pyarrow.Array`

#### Type behavior

- Float input (`float32`/`float64`) is cast to `float64`
- Non-float input is cast to `int64`
- Callable return values are converted back to the active numeric lane:
  - integer lane -> `i64` conversion
  - float lane -> `f64` conversion

## Callable constraints

Workers import by module + function name inside sub-interpreters. Because of that:

- lambdas are rejected
- local/nested functions are rejected
- functions defined directly in `__main__` are rejected

Put callables in importable modules (for example `tasks.py`) and import them in your entrypoint.

## Error model

| Exception | Condition |
| --- | --- |
| `TypeError` | `func` is not callable |
| `ValueError` | lambda/local function/`__main__` function used |
| `TypeError` | input cannot be cast to supported numeric Arrow type |
| `RuntimeError` | execution/import failure in worker sub-interpreters |

If any worker fails, the whole call fails and no partial result is returned.

## Architecture

### Python layer (`gilmap/__init__.py`)

1. Validates callable constraints
2. Converts/casts input to Arrow (`int64` or `float64`)
3. Calls Rust extension entrypoint `execute(module_name, func_name, array, sys.path)`
4. Converts result to list when original input was not Arrow
5. Registers `shutdown_workers` with `atexit` for clean worker teardown

### Rust layer (`src/lib.rs`)

1. Lazily initializes a global worker pool (`OnceLock`) on first call
2. Starts one thread per `available_parallelism()`
3. Creates one Python sub-interpreter per worker thread
4. Receives chunked tasks over a shared queue
5. Caches imported function objects per worker (`(module_name, func_name)` key)
6. Extends worker `sys.path` from caller-provided entries
7. Executes function for each value in the chunk
8. Signals completion via `Condvar`
9. Reassembles chunked output into Arrow array and returns to Python

## Worker lifecycle and shutdown

- Worker threads/sub-interpreters are long-lived after first use
- They are reused across `gilmap.map` calls
- `shutdown_workers` sends one shutdown message per worker and joins threads
- `shutdown_workers` is automatically called at process exit via `atexit`

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
On `float_math` at N=1,000,000, **gilmap** (gilmap_arrow@arrow) is **77.58× faster** than the best non-gilmap runner (`numba`) — measured on Apple M3 Max.

**Summary across 6 workloads:**

| workload | best gilmap variant | speedup vs runner-up | runner-up |
|---|---|---|---|
| float_math N=1,000,000 | gilmap_arrow@arrow | 77.58× | numba |
| float_math N=100,000 | gilmap_arrow@arrow | 63.70× | numba |
| quick_collatz N=1,000,000 | gilmap_arrow@arrow | 52.02× | numba |
| quick_collatz N=100,000 | gilmap_arrow@arrow | 44.65× | numba |
| quick_collatz N=10,000 | gilmap_arrow@arrow | 21.57× | numpy_vec |
| ⚠ mandelbrot_iters N=1,000 | gilmap_arrow@arrow | 0.22× (we lose) | numba |
| ⚠ count_primes N=512 | gilmap_arrow@arrow | 0.89× (we lose) | joblib |
| ⚠ heavy_collatz N=64 | gilmap_arrow@arrow | 0.95× (we lose) | cf_process |

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
├── src/lib.rs
├── gilmap/__init__.py
├── tests/
│   ├── tasks.py
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
