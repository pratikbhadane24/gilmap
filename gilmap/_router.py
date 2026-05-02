"""Aggressive auto-router: pick the fastest backend for a given user callable.

Detection order (fastest first; first match wins):
    1. numba @njit / cfunc / ctypes / cffi → call native pointer in worker loop
    2. Arrow-kernel-shape AST → lower to pyarrow.compute (one C call for whole array)
    3. Cranelift JIT (whitelisted AST subset) — SKELETON, falls through for now
    4. Sub-interpreter path (current Rust pool)

The router caches its decision per-callable on a weakref so repeat calls are free.

Public surface lives in gilmap/__init__.py via:
    gilmap.map(func, iter, *, debug=False)        # transparent dispatch
    gilmap.explain(func)                           # introspect chosen backend
"""

from __future__ import annotations

import ast
import inspect
import os
import textwrap
import weakref
from dataclasses import dataclass
from typing import Any, Callable

import pyarrow as pa
import pyarrow.compute as pc

# ----------------------------------------------------------------------
# Decision record
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class BackendDecision:
    backend: str  # one of: "numba_native", "arrow_kernel", "jit", "subinterp"
    reason: str
    # Optional callable that implements the fast path. None means the caller
    # routes to the default sub-interp executor.
    fast_path: Callable[[Callable, pa.Array], pa.Array] | None = None


# ----------------------------------------------------------------------
# Caching
# ----------------------------------------------------------------------

# WeakValueDictionary keyed by id(func). We use a plain dict keyed by id and
# attach a weak-finalizer to clear it when the function is GC'd, so repeat
# calls hit O(1) without leaking decisions for short-lived lambdas (which we
# reject anyway, but defense in depth).
_DECISION_CACHE: dict[int, BackendDecision] = {}
_FINALIZERS: dict[int, weakref.ref] = {}


def _cache_decision(func: Callable, decision: BackendDecision) -> BackendDecision:
    fid = id(func)
    _DECISION_CACHE[fid] = decision
    try:
        _FINALIZERS[fid] = weakref.ref(func, lambda _: _evict(fid))
    except TypeError:
        # Non-weak-referenceable callable (builtin, C function). Don't cache.
        _DECISION_CACHE.pop(fid, None)
    return decision


def _evict(fid: int) -> None:
    _DECISION_CACHE.pop(fid, None)
    _FINALIZERS.pop(fid, None)


def _cached_decision(func: Callable) -> BackendDecision | None:
    return _DECISION_CACHE.get(id(func))


# ----------------------------------------------------------------------
# Detection: numba / cfunc / ctypes / cffi
# ----------------------------------------------------------------------

def _detect_numba(func: Callable) -> BackendDecision | None:
    """Detect numba-decorated callables and route to a vectorized form.

    Strategy: numba `@njit` callables expose `nopython_signatures` and a
    `get_compile_result` helper. The reliable way to apply them across a
    pyarrow array is to convert to a numpy view and call func element-wise
    — numba's dispatcher caches its specialization, so per-call cost stays
    in the tens of nanoseconds (vs hundreds for plain CPython).

    A future move can pull `func.address` from a `@cfunc` and call the
    raw extern "C" pointer from Rust; left as TODO for now because the
    safe fast path (numpy round-trip) is already a big win.
    """
    is_njit = (
        hasattr(func, "nopython_signatures")
        or getattr(func, "_is_njit", False)
        or getattr(func.__class__, "__module__", "").startswith("numba")
    )
    is_cfunc = (
        hasattr(func, "address") and callable(func) and isinstance(getattr(func, "address", None), int)
    )
    if not (is_njit or is_cfunc):
        return None

    def fast(f: Callable, arr: pa.Array) -> pa.Array:
        # Convert to numpy (zero-copy when dtype matches) and call f over
        # each element. With numba's specialization cached this is ~10-20ns
        # per element vs CPython's ~150ns per call.
        import numpy as np

        np_in = arr.to_numpy(zero_copy_only=False)
        out = np.empty_like(np_in)
        for i in range(np_in.shape[0]):
            out[i] = f(np_in[i])
        return pa.array(out)

    label = "numba_njit" if is_njit else "cfunc"
    return BackendDecision(
        backend="numba_native",
        reason=f"detected {label} via attribute introspection",
        fast_path=fast,
    )


# ----------------------------------------------------------------------
# Detection: Arrow-kernel-shape AST
# ----------------------------------------------------------------------

# Maps Python AST node-class to (pyarrow.compute fn, arity)
# Single-expression functions of form `lambda x: x*c + d` (or any chain of
# the supported binops over `x` and constants) lower to one or more
# pyarrow.compute calls — the whole array is processed in a single C call.
_BINOPS = {
    ast.Add: pc.add,
    ast.Sub: pc.subtract,
    ast.Mult: pc.multiply,
    ast.Div: pc.divide,
    ast.Mod: pc.divide,  # NOTE: pc.divide ≠ python %; we'll explicitly skip Mod below
    ast.Pow: pc.power,
    ast.BitAnd: pc.bit_wise_and,
    ast.BitOr: pc.bit_wise_or,
    ast.BitXor: pc.bit_wise_xor,
}

_UNARYOPS = {
    ast.USub: pc.negate,
    ast.UAdd: lambda x: x,
    ast.Invert: pc.bit_wise_not,
}

# math.* mapping → pyarrow.compute
_MATH_CALLS = {
    "sqrt": pc.sqrt,
    "abs": pc.abs,
    "exp": pc.exp,
    "log": pc.ln,
    "sin": pc.sin,
    "cos": pc.cos,
    "tan": pc.tan,
    "floor": pc.floor,
    "ceil": pc.ceil,
    "round": pc.round,
}


def _get_source(func: Callable) -> str | None:
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        return None
    return textwrap.dedent(src)


def _func_body_expr(func: Callable) -> tuple[ast.expr, str] | None:
    """If `func` body is a single `return <expr>`, return (expr, param_name).
    None otherwise.
    """
    src = _get_source(func)
    if src is None:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    if not tree.body:
        return None
    # `inspect.getsource(lambda)` returns the source line including any
    # surrounding assignment (`f = lambda x: ...`). Walk the tree to find
    # the FunctionDef or Lambda we actually care about.
    fn_node: ast.FunctionDef | ast.Lambda | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Lambda)):
            fn_node = node
            break
    if fn_node is None:
        return None
    if isinstance(fn_node, ast.Lambda):
        params = fn_node.args.args
        body = fn_node.body
    else:
        params = fn_node.args.args
        # Strip a leading docstring (Expr(Constant(str))) so functions with
        # a single `return <expr>` after their docstring still lower.
        stmts = fn_node.body
        if (
            stmts
            and isinstance(stmts[0], ast.Expr)
            and isinstance(stmts[0].value, ast.Constant)
            and isinstance(stmts[0].value.value, str)
        ):
            stmts = stmts[1:]
        if len(stmts) != 1 or not isinstance(stmts[0], ast.Return):
            return None
        body = stmts[0].value
    if len(params) != 1:
        return None
    if body is None:
        return None
    return body, params[0].arg


def _detect_arrow_kernel(func: Callable) -> BackendDecision | None:
    parsed = _func_body_expr(func)
    if parsed is None:
        return None
    body, param_name = parsed

    def lower(node: ast.expr) -> Callable[[pa.Array], pa.Array | Any] | None:
        """Compile an AST expr into a closure that takes the input array
        and returns a pa.Array (or scalar) using pyarrow.compute kernels.
        Returns None if any node is unsupported."""
        if isinstance(node, ast.Name):
            if node.id == param_name:
                return lambda arr: arr
            return None  # unknown free variable
        if isinstance(node, ast.Constant):
            return lambda _arr, _v=node.value: _v
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARYOPS:
                return None
            op = _UNARYOPS[type(node.op)]
            inner = lower(node.operand)
            if inner is None:
                return None
            return lambda arr: op(inner(arr))
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Mod):
                return None  # pc has no exact py-% semantics for negatives
            if type(node.op) not in _BINOPS:
                return None
            op = _BINOPS[type(node.op)]
            left = lower(node.left)
            right = lower(node.right)
            if left is None or right is None:
                return None
            return lambda arr: op(left(arr), right(arr))
        if isinstance(node, ast.Call):
            # Only `math.<name>(arg)` form
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "math"
                and node.func.attr in _MATH_CALLS
                and len(node.args) == 1
                and not node.keywords
            ):
                kernel = _MATH_CALLS[node.func.attr]
                arg = lower(node.args[0])
                if arg is None:
                    return None
                return lambda arr: kernel(arg(arr))
            return None
        if isinstance(node, ast.IfExp):
            # x*a if cond else x*b — supported via pc.if_else
            cond = lower(node.test)
            yes = lower(node.body)
            no = lower(node.orelse)
            if cond is None or yes is None or no is None:
                return None
            return lambda arr: pc.if_else(cond(arr), yes(arr), no(arr))
        if isinstance(node, ast.Compare):
            # Single-op compare only (a < b). Chained (a<b<c) not supported.
            if len(node.ops) != 1 or len(node.comparators) != 1:
                return None
            cmpmap = {
                ast.Lt: pc.less,
                ast.LtE: pc.less_equal,
                ast.Gt: pc.greater,
                ast.GtE: pc.greater_equal,
                ast.Eq: pc.equal,
                ast.NotEq: pc.not_equal,
            }
            if type(node.ops[0]) not in cmpmap:
                return None
            op = cmpmap[type(node.ops[0])]
            left = lower(node.left)
            right = lower(node.comparators[0])
            if left is None or right is None:
                return None
            return lambda arr: op(left(arr), right(arr))
        return None

    compiled = lower(body)
    if compiled is None:
        return None

    def fast(_f: Callable, arr: pa.Array) -> pa.Array:
        result = compiled(arr)
        # `result` may be a scalar if the body was a constant — wrap.
        if not isinstance(result, pa.Array):
            result = pa.array([result] * len(arr))
        return result

    return BackendDecision(
        backend="arrow_kernel",
        reason="body lowers to pyarrow.compute kernel chain",
        fast_path=fast,
    )


# ----------------------------------------------------------------------
# Detection: Cranelift JIT (skeleton)
# ----------------------------------------------------------------------

def _detect_jit(_func: Callable) -> BackendDecision | None:
    """Cranelift JIT path. Skeleton: not yet wired to Rust.

    When implemented, this detects a wider AST whitelist (loops, locals,
    conditionals over numerics) and lowers to Cranelift IR with std::simd.
    For now we always return None; the router falls through to subinterp.
    """
    return None


# ----------------------------------------------------------------------
# Top-level routing
# ----------------------------------------------------------------------

DETECTORS: list[tuple[str, Callable[[Callable], BackendDecision | None]]] = [
    # arrow_kernel first: a body that lowers to pyarrow.compute is faster
    # than numba's per-scalar dispatcher even when the user decorated with
    # @njit. Decorator presence does not imply numba's path is fastest.
    ("arrow_kernel", _detect_arrow_kernel),
    ("numba_native", _detect_numba),
    ("jit", _detect_jit),
]


def decide(func: Callable) -> BackendDecision:
    cached = _cached_decision(func)
    if cached is not None:
        return cached
    for _name, detector in DETECTORS:
        decision = detector(func)
        if decision is not None:
            return _cache_decision(func, decision)
    # Default: sub-interpreter path (the original gilmap behavior).
    fallback = BackendDecision(
        backend="subinterp",
        reason="no faster backend matched",
        fast_path=None,
    )
    return _cache_decision(func, fallback)


def reset_cache() -> None:
    """Test hook — clear cached decisions."""
    _DECISION_CACHE.clear()
    _FINALIZERS.clear()


def debug_enabled() -> bool:
    return os.environ.get("GILMAP_DEBUG", "").strip() not in ("", "0", "false", "False")
