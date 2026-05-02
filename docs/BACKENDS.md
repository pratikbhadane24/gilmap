# gilmap backends

`gilmap.map(func, iter)` consults an auto-router that picks one of several
backends per call. The decision is cached per-callable (weakref-keyed) so
repeat calls are O(1).

Use `gilmap.explain(func)` to see which backend will be selected.
Set `GILMAP_DEBUG=1` (or pass `debug=True`) to print the chosen backend on
each call.

## Detection order

The router walks detectors in priority order. First match wins.

| # | Backend | Selected when | Where work runs |
|---|---|---|---|
| 1 | `numba_native` | `func` exposes `nopython_signatures`, `_is_njit`, lives in a `numba.*` module, or has an integer `address` (cfunc) | numpy round-trip + cached numba dispatcher; no CPython per element |
| 2 | `arrow_kernel` | `func` body is a single `return <expr>` (or lambda body) where `<expr>` is composed of supported binops/unops/`math.*`/comparisons over the input parameter and constants | one or more `pyarrow.compute` calls — whole array in a single C call |
| 3 | `jit` (skeleton) | wider AST whitelist (loops, locals, conditionals over numerics) — **not yet implemented** | will lower to Cranelift IR with `std::simd` lanes |
| 4 | `subinterp` | nothing else matches | per-worker Python sub-interpreter pool (the original gilmap engine, with all P1 Tier-1+Tier-2 fixes) |

## What each backend assumes

### `numba_native`
- `func` must be safe to call repeatedly with scalar inputs; the dispatcher
  caches the specialization. We do not currently extract the raw native
  pointer (TODO), so the gain comes from skipping CPython's call overhead
  via numba's specialized callable.
- For `@cfunc`, we go through the same numpy round-trip; future work pulls
  `func.address` and calls the `extern "C"` pointer from Rust.

### `arrow_kernel`
- Lowered AST node set today: `Constant`, `Name(load)`, `BinOp` (`+`, `-`,
  `*`, `/`, `**`, `&`, `|`, `^`), `UnaryOp` (`-`, `+`, `~`), `Compare`
  (single comparator), `IfExp`, and `Call` for `math.{sqrt,abs,exp,log,sin,
  cos,tan,floor,ceil,round}` only.
- Excluded: `Mod` (Python `%` ≠ `pyarrow.compute.divide`'s integer form for
  negatives), arbitrary attribute access, `len`, comprehensions, etc.
- Result dtype is whatever pyarrow.compute returns; we coerce back to the
  input lane (`int64`/`float64`) when possible. Some operations (e.g.
  `divide` on int inputs) inherently return float — we preserve the kernel
  dtype rather than lossy-cast.

### `jit` (skeleton)
Reserved for the Cranelift JIT path. `_detect_jit` always returns None
today. When implemented:
- Whitelist will widen to counted loops, local assignments, and
  conditionals.
- Compile path: Python AST → typed IR (Rust struct) → Cranelift IR →
  native function called from worker chunks with SIMD lanes.

### `subinterp`
- The function must be importable from a sub-interpreter — i.e.
  module-level, no lambdas, no `<locals>` qualname, not in `__main__`.
- This is the only path that requires those constraints. The router's
  fast paths above are stricter on body shape but looser on identity, so
  lambdas and `__main__` functions can still run as long as their bodies
  are routable.
- On a free-threaded build (`Py_GIL_DISABLED=1`), the sub-interpreter pool
  is unavailable and reaching this branch raises a targeted `RuntimeError`.
  Free-threaded support for sub-interpreter parallelism is a future move.

## Examples

```python
import gilmap, math
from numba import njit

# numba_native
@njit(cache=True)
def kernel(x): return x * x + 1
gilmap.explain(kernel)
# {'backend': 'numba_native', 'reason': '...', 'has_fast_path': True}

# arrow_kernel
gilmap.explain(lambda x: math.sqrt(x) * 2.0)
# {'backend': 'arrow_kernel', ..., 'has_fast_path': True}

# subinterp (multi-statement body)
def heavy(x):
    s = 0
    for i in range(x):
        s += i * x
    return s
gilmap.explain(heavy)
# {'backend': 'subinterp', ..., 'has_fast_path': False}
```

## Honesty

The auto-router is opinionated: it routes to whichever path tends to be
fastest *in our benchmark suite*. It does not try every backend on every
call. If you want to force a specific path, the right hooks are
(planned, not implemented yet) `gilmap.compile(func, backend=...)`.

Where the router *guesses wrong* — e.g. an arrow-kernel-shape body whose
computation is so light that the per-batch dispatch costs more than the
per-element CPython loop — file an issue with `gilmap.explain(func)`'s
output and a benchmark, and we'll widen the heuristics.
