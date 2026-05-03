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
| 3 | `jit` | single-`return <expr>` body of binops/unops/compare/ternary/`math.*` calls — same shape as `arrow_kernel` but compiled to native via Cranelift. Activates when arrow_kernel rejects (e.g. `%` on int lane) | per-element loop in JITed native code |
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

### `jit`
- v1 whitelist (shared with `arrow_kernel`): single `return <expr>` body
  composed of `Constant`, `Name(load)`, `BinOp` (incl. `%`, `//`, `**`),
  `UnaryOp`, `Compare`, `IfExpr`, and `Call` for `math.{sqrt,abs,exp,log,
  sin,cos,tan,floor,ceil}`.
- Compile path: Python AST → typed IR (`src/ast_ir.rs`) → JSON →
  Cranelift IR → native `extern "C" fn(*const T, *mut T, len)` →
  `_gilmap.jit_apply`.
- Stable kernel hash from the IR JSON; recompiled only on cache miss.
- Current routing: arrow_kernel runs first (its pyarrow.compute kernels
  are hand-tuned SIMD and beat scalar-loop native code on the trivially
  vectorizable shapes both backends share). JIT v1 actually fires for
  shapes arrow_kernel rejects — most notably integer `%`, which matches
  `quick_collatz`-style ternaries: `n // 2 if n % 2 == 0 else 3 * n + 1`.
- v1 limitations (P5b will lift): no local assigns, no counted for-loops,
  no if-statements with early return. That set is what `mandelbrot_iters`
  needs; until P5b lands it stays on `subinterp`.
- Cranelift-specific quirks: integer `%` uses `srem` (C-style signed
  remainder), which differs from Python's `%` for negative inputs. Don't
  route negative-input `%` through JIT until we add a Python-semantics
  helper (`x - floordiv(x, y) * y`).

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

# jit (% rejected by arrow_kernel; JIT compiles to native srem)
def collatz_step(n):
    return n // 2 if n % 2 == 0 else 3 * n + 1
gilmap.explain(collatz_step)
# {'backend': 'jit', ..., 'has_fast_path': True}

# subinterp (multi-statement body — needs P5b)
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
