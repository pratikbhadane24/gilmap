# hyperfunctions

`hyperfunctions` is a Python + Rust parallel map runtime for numeric, single-argument, module-level Python functions.

It combines:

- Rust worker threads for CPU parallelism
- Python sub-interpreters (one per worker thread)
- Apache Arrow arrays for efficient Python <-> Rust transfer

The public API is one function: `hyperfunctions.map`.

## What it is optimized for

`hyperfunctions.map` is best for **CPU-bound** functions where each element does enough work to amortize scheduling/conversion overhead.

It is usually a poor fit for:

- tiny per-element work (overhead dominates)
- I/O-bound callables
- lambdas/local functions/not-importable functions

## Requirements

- Python `>= 3.12`
- Rust toolchain (`cargo`, `rustc`)
- `maturin`
- `pyarrow`

## Install (local development)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip maturin pyarrow pytest
maturin develop
```

`maturin develop` builds and installs the `_hyperfunctions` extension for the active environment.

## Quick start

### Integer input

```python
# tasks.py
def square(x: int) -> int:
    return x * x
```

```python
# app.py
import hyperfunctions
from tasks import square

out = hyperfunctions.map(square, [1, 2, 3, 4])
print(out)  # [1, 4, 9, 16]
```

### Float input

```python
def affine(x: float) -> float:
    return x * 1.5 + 0.5

print(hyperfunctions.map(affine, [1.0, 2.0, 3.0]))
```

### Arrow input (Arrow output)

```python
import pyarrow as pa
import hyperfunctions
from tasks import square

arr = pa.array([1, 2, 3, 4], type=pa.int64())
out = hyperfunctions.map(square, arr)
print(type(out))  # pyarrow.Array
```

## API reference

### `hyperfunctions.map(func, iterable) -> list | pyarrow.Array`

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

### Python layer (`hyperfunctions/__init__.py`)

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
- They are reused across `hyperfunctions.map` calls
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

Benchmark harnesses are included in `tests/test_heavy.py` and `tests/test_parallel.py`.

### Benchmark workloads in repo

| Workload | Function(s) | Goal |
| --- | --- | --- |
| Prime counting | `count_primes` | CPU-heavy integer compute |
| Heavy collatz | `heavy_collatz` | Long iterative integer compute |
| Overhead stress | `quick_collatz` over ~1M values | Measures framework overhead on light compute |
| Float pipeline | `float_math` over ~1M values | Float path behavior + Arrow/list input comparison |

### Run benchmark suite

```bash
source .venv/bin/activate
python tests/test_heavy.py
```

The script prints timings for:

- standard `map`
- `multiprocessing.Pool.map`
- `hyperfunctions.map` with list input
- `hyperfunctions.map` with Arrow input (for overhead/float cases)

### Interpreting results

- Expect strongest wins on CPU-heavy tasks with enough per-element work.
- For tiny operations, plain `map` can be faster due to lower overhead.
- Arrow input often reduces conversion overhead versus list input for large arrays.
- Compare against `multiprocessing` on your target machine; winner depends on workload shape and IPC cost.

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
├── hyperfunctions/__init__.py
└── tests/
    ├── tasks.py
    ├── test_parallel.py
    ├── test_safety.py
    └── test_heavy.py
```

## CI and packaging

The GitHub Actions workflow builds wheels/sdists for multiple platforms using maturin.
