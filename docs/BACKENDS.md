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
| 1 | `arrow_kernel` | `func` body is a single `return <expr>` (or lambda body) where `<expr>` is composed of supported binops/unops/`math.*`/comparisons over the input parameter and constants | one or more `pyarrow.compute` calls — whole array in a single C call |
| 2 | `numba_cfunc` | `func` is a `@cfunc(types.<T>(types.<T>))` with a same-dtype scalar signature (i64→i64 or f64→f64) | raw `extern "C"` pointer (`func.address`) called per element from Rust, chunked across rayon workers; no Python C-API per element |
| 3 | `numba_native` | `func` exposes `nopython_signatures`, `_is_njit`, lives in a `numba.*` module, or is a `@cfunc` whose signature isn't routable to `numba_cfunc` (e.g. mixed-dtype) | numpy round-trip + cached numba dispatcher; no CPython per element |
| 4 | `jit` | numeric body of binops/unops/compare/ternary/`math.*`, plus local `Assign`/`AugAssign`, counted `for i in range(...)`, `while`, `if`/`else` (incl. early return), `break`, `continue`. Activates for shapes arrow_kernel rejects (e.g. `%` on int lane, multi-statement bodies) | per-element loop in JITed native code |
| 5 | `subinterp` | nothing else matches | per-worker Python sub-interpreter pool (the original gilmap engine, with all P1 Tier-1+Tier-2 fixes) |

> Detection order note: `arrow_kernel` runs before `numba_native` because a body that lowers to pyarrow.compute SIMD kernels beats numba's per-scalar dispatcher even when the user decorated with `@njit`. Decorator presence does not imply numba's path is fastest.

## What each backend assumes

### `numba_cfunc`
- Activates only for `@cfunc(types.<T>(types.<T>))` decorators where input
  and output dtype both resolve to `int64` or `float64` (same on both
  sides). Detection reads `func._sig.args` and `func._sig.return_type`.
- `func.address` (a stable process-lifetime `extern "C" fn(T) -> T`
  pointer) is captured at decide-time and called from Rust per element via
  `_gilmap.cfunc_apply` (`src/cfunc_dispatch.rs`). The dispatcher chunks
  across rayon workers using the same `num_threads * 4` heuristic the JIT
  path uses.
- Skips the numpy round-trip and the Python C-API entirely on the hot
  loop. Approximate per-element cost: a single C call (~1ns) vs the
  legacy `numba_native` path's ~10-20ns per element.
- Falls through to `numba_native` when the signature is mixed-dtype
  (e.g. `int64(float64)`), multi-arg, or unreadable.

### `numba_native`
- `func` must be safe to call repeatedly with scalar inputs; the dispatcher
  caches the specialization. The gain comes from skipping CPython's call
  overhead via numba's specialized callable.
- For `@cfunc`, this is the fallback when the signature isn't routable to
  `numba_cfunc` (e.g. mixed-dtype). For `@njit`, it's the only numba path.

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
- Supported AST set:
  - Expressions: `Constant`, `Name(load)`, `BinOp` (incl. `%`, `//`, `**`),
    `UnaryOp`, `Compare`, `IfExpr`, and `Call` for `math.{sqrt,abs,fabs,exp,
    log,sin,cos,tan,floor,ceil}`.
  - Statements: `Return`, `Assign`/`AugAssign` (per-local `i64`/`f64` type
    inference), counted `for i in range(...)`, `while`, `if`/`else` (incl.
    `if cond: return X` early return), `break`, `continue`.
- **SIMD vectorization**: kernels whose body is a single `return <expr>` of
  pure arithmetic / compare / ternary / cast (no locals, no math calls, no
  integer mod/div/pow) compile to a 128-bit-vector main loop (`F64X2` or
  `I64X2`, 2 elements per iteration) plus a scalar tail. Cranelift lowers
  to native SSE2/NEON at codegen time. Anything outside that pure-expr
  shape — math calls, locals, loops, integer modulo — uses the unchanged
  scalar codegen path.
- Compile path: Python AST → typed IR (`src/ast_ir.rs`) → JSON →
  Cranelift IR → native `extern "C" fn(*const T, *mut T, len)` →
  `_gilmap.jit_apply`.
- Stable kernel hash from the IR JSON; recompiled only on cache miss.
- Type inference: per-local dtype tag; `BinOp`/`Compare`/`IfExpr` lift to
  f64 when either operand is f64, else stay i64. Output dtype is whatever
  the first reachable `Return` produces; subsequent returns must agree.
- Current routing: arrow_kernel runs first (its pyarrow.compute kernels
  are hand-tuned SIMD and beat scalar-loop native code on the trivially
  vectorizable shapes both backends share). JIT fires for shapes
  arrow_kernel rejects — most notably integer `%`, multi-statement bodies,
  and counted loops (e.g. `mandelbrot_iters`).
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
- On a free-threaded build (`Py_GIL_DISABLED=1`), this backend transparently
  swaps its parallelism primitive: instead of PEP 684 sub-interpreters with
  own-GIL (incompatible with `Py_GIL_DISABLED`), workers share the main
  interpreter and run on a rayon thread pool (`_gilmap.execute_rayon`). The
  importability rules are unchanged; the chunking math (`pick_chunk_size`)
  and per-element FFI loop (`src/per_element.rs`) are shared between the two
  paths. Result: free-threaded users hit a real parallel executor instead of
  a `RuntimeError`.

## Examples

```python
import gilmap, math
from numba import njit, cfunc, types

# numba_cfunc — raw extern "C" pointer, no numpy round-trip
@cfunc(types.int64(types.int64))
def cfunc_kernel(n):
    s = 0
    for i in range(8):
        s += (n + i) % 7
    return s
gilmap.explain(cfunc_kernel)
# {'backend': 'numba_cfunc', 'reason': '...', 'has_fast_path': True}

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

# jit (multi-statement body with counted for-loop)
def heavy(x):
    s = 0
    for i in range(x):
        s += i * x
    return s
gilmap.explain(heavy)
# {'backend': 'jit', ..., 'has_fast_path': True}

# subinterp (uses an unsupported call — falls through)
def with_print(x):
    print(x)
    return x
gilmap.explain(with_print)
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
