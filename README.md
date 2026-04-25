# hyperfunctions

`hyperfunctions` is a hybrid Python + Rust library that runs Python callables over integer data in parallel using:

- Rust threads for CPU-level concurrency
- Python sub-interpreters (Python 3.12+) so each worker can execute Python code independently
- Apache Arrow for efficient data movement between Python and Rust

At the Python level, the library exposes one function: `hyperfunctions.map`.

## What problem this solves

`hyperfunctions.map` is designed for CPU-bound, pure-ish functions that are expensive enough to benefit from parallel execution. It provides a familiar `map`-style API while offloading scheduling and execution to a Rust backend.

## Current feature scope

The current implementation is intentionally narrow:

- Input values are limited to integers (`int64` in Arrow)
- The mapped function must be importable from a real module
- The mapped function must return an integer (`i64`-extractable value)
- Output is returned as a Python `list[int]`

This keeps cross-interpreter execution predictable and minimizes serialization complexity.

## Installation

### Prerequisites

- Python `>= 3.12`
- Rust toolchain (`rustup`, `cargo`)
- `maturin` (build backend)
- `pyarrow`

### Local development install

```bash
# from repository root
python -m venv .venv
source .venv/bin/activate
pip install -U pip maturin pyarrow pytest
maturin develop
```

`maturin develop` compiles the Rust extension (`_hyperfunctions`) and installs the Python package in editable mode.

## Quick start

`hyperfunctions.map` requires a module-level function (not a lambda or locally nested function).

```python
# tasks.py
def square(x: int) -> int:
    return x * x
```

```python
# app.py
import hyperfunctions
from tasks import square

result = hyperfunctions.map(square, [1, 2, 3, 4])
print(result)  # [1, 4, 9, 16]
```

## API reference

### `hyperfunctions.map(func, iterable) -> list[int]`

Executes `func` over `iterable` in parallel.

### Parameters

- `func`: callable taking one integer argument and returning an integer
- `iterable`: either:
  - a Python iterable of integer-like values, or
  - a `pyarrow.Array` that is already `int64` (or castable to it)

### Returns

- `list[int]` containing one result per input element, in input order

### Raised exceptions

| Exception | When it is raised |
| --- | --- |
| `TypeError` | `func` is not callable |
| `ValueError` | `func` is a lambda |
| `ValueError` | `func` is a local/nested function |
| `ValueError` | `func.__module__ == "__main__"` |
| `TypeError` | input cannot be represented/cast as `int64` |
| `RuntimeError` | worker thread/interpreter execution fails |

## Function constraints and why they exist

The backend imports your function by module and name inside each sub-interpreter:

1. It reads `func.__module__` and `func.__name__`
2. Worker interpreters import that module
3. Workers resolve and invoke the function by attribute lookup

Because of that:

- **No lambda** (`__name__ == "<lambda>"`)
- **No local function** (`"<locals>"` appears in `__qualname__`)
- **No direct `__main__` functions** (sub-interpreters cannot reliably import your script entrypoint module by that name)

If needed, move your function into a separate module (for example `tasks.py`) and import it in your main script.

## Data model and type conversion

The Python wrapper normalizes input into Arrow:

1. Non-Arrow iterables are converted with `pa.array(...)`
2. Arrays are cast to `pa.int64()` when possible
3. Non-castable input raises `TypeError`

The Rust backend:

1. Converts PyArrow input to Arrow `Int64Array`
2. Splits values into chunks based on available parallelism
3. Executes chunk work in Rust threads + Python sub-interpreters
4. Builds an Arrow `Int64Array` from results

The Python wrapper converts the resulting Arrow array back to a plain list.

## Architecture

### Python layer (`hyperfunctions/__init__.py`)

- Validates callable restrictions
- Converts/casts input to Arrow `int64`
- Passes module name, function name, input array, and `sys.path` into Rust
- Converts result array to `list[int]`

### Rust extension (`src/lib.rs`)

- Exposes `_hyperfunctions.execute(...)` via PyO3
- Deserializes PyArrow input into Arrow array
- Determines worker count using `available_parallelism()`
- Uses `std::thread::scope` for chunk workers
- Creates one Python sub-interpreter per worker with `Py_NewInterpreterFromConfig`
- Appends incoming `sys.path` entries in each worker interpreter
- Imports module and calls target function for each value
- Ends each sub-interpreter with `Py_EndInterpreter`
- Returns result array to Python

## Repository layout

```text
.
├── Cargo.toml                  # Rust crate metadata and dependencies
├── pyproject.toml              # Python package metadata (maturin backend)
├── src/lib.rs                  # Rust execution engine exposed to Python
├── hyperfunctions/__init__.py  # Public Python API wrapper
├── tests/
│   ├── tasks.py                # Test task functions
│   ├── test_parallel.py        # Basic correctness/perf comparison test
│   ├── test_safety.py          # Validation and error propagation tests
│   └── test_heavy.py           # Manual benchmark-style script
└── .github/workflows/CI.yml    # Multi-platform wheel/sdist build pipeline
```

## Running tests

Use the project virtual environment when available:

```bash
.venv/bin/python -m pytest -q
```

Rust compile check/tests:

```bash
cargo test
```

## Performance notes

- Best for CPU-heavy functions where work per element dominates overhead.
- For very small or trivial functions, setup and conversion overhead can outweigh benefits.
- Performance characteristics depend on:
  - number of CPU cores
  - cost variance across input elements
  - module import/function call overhead in worker interpreters

## Error propagation model

If a mapped Python function raises inside a worker interpreter, the backend returns a `RuntimeError` to the caller with a worker-thread error message. This fails the whole `map` call; partial results are not returned.

## CI and packaging

The GitHub Actions workflow (generated by maturin) builds wheels for Linux, musllinux, Windows, and macOS targets, plus source distributions. Tagged builds are configured for publication.

## Known limitations

- Integer-only input/output (`int64`)
- Single-argument mapped function shape
- No lambda/local/`__main__` functions
- Return values must be integer-extractable in Rust (`i64`)

These constraints reflect the current implementation and can be expanded in future versions.
