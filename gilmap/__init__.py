"""Public Python API for gilmap.

`gilmap.map(func, iter)` is a parallel map. It runs an aggressive auto-router
that picks the fastest backend for the user callable:

    1. numba @njit / cfunc / ctypes  → call native pointer (router fast path)
    2. Arrow-kernel-shape AST        → lower to pyarrow.compute (one C call)
    3. Cranelift JIT (whitelist)     → JIT scalar Python to native (skeleton)
    4. Sub-interpreter pool          → fallback for arbitrary module-level fns

The router is silent by default. Set ``GILMAP_DEBUG=1`` or pass
``debug=True`` to print the chosen backend per call. Use
``gilmap.explain(func)`` for programmatic introspection.
"""

from collections.abc import Callable, Iterable
import atexit
import os
import sys
import sysconfig

import pyarrow as pa

from _gilmap import execute, shutdown_workers

from . import _router

# Free-threaded (PEP 703) builds: gilmap's sub-interpreter pool relies on
# Py_NewInterpreterFromConfig with PyInterpreterConfig_OWN_GIL, which is not
# meaningful when the GIL is disabled globally. We don't refuse to load
# anymore — instead, the router still picks the fast paths (numba native,
# Arrow kernels, JIT) which work fine on free-threaded builds. The
# sub-interpreter fallback is gated; if it's needed on a free-threaded build
# we raise a targeted error at dispatch time.
_FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))

atexit.register(shutdown_workers)


def explain(func: Callable) -> dict:
    """Return the router's decision for `func` without executing it.

    Useful for users to verify which backend will be selected, and for the
    benchmark harness to assert backend dispatch in tests.

    Returns a dict with keys: ``backend``, ``reason``, ``has_fast_path``.
    """
    if not callable(func):
        raise TypeError("explain() requires a callable")
    decision = _router.decide(func)
    return {
        "backend": decision.backend,
        "reason": decision.reason,
        "has_fast_path": decision.fast_path is not None,
    }


def _validate_for_subinterp(func: Callable) -> tuple[str, str]:
    """Importability checks for the sub-interpreter fallback. Skipped when
    the router picks a fast path that doesn't import from worker sub-interps.
    """
    func_name = getattr(func, "__name__", "")
    if func_name == "<lambda>":
        raise ValueError(
            "gilmap.map does not support lambda functions on the sub-interpreter "
            "fallback. Define the function at module level, or write it in a form "
            "the router can lower to a fast path (Arrow kernel / numba)."
        )
    if "<locals>" in getattr(func, "__qualname__", ""):
        raise ValueError(
            "gilmap.map does not support local/nested functions on the "
            "sub-interpreter fallback. Move it to module scope."
        )
    module_name = getattr(func, "__module__", "")
    if module_name == "__main__":
        raise ValueError(
            "gilmap.map cannot execute functions defined in the __main__ script "
            "directly on the sub-interpreter fallback. Define your function in a "
            "separate module and import it."
        )
    return module_name, func_name


def _to_arrow(iterable: Iterable | pa.Array) -> tuple[pa.Array, bool]:
    """Convert input to a pyarrow.Array suitable for the engine. Returns
    (array, return_list) where return_list signals whether the user passed
    a Python iterable (so we should hand back a list)."""
    return_list = False
    if not isinstance(iterable, pa.Array):
        iterable = pa.array(iterable)
        return_list = True

    # Tier-1: skip cast when dtype already matches.
    arr_type = iterable.type
    if pa.types.is_float64(arr_type) or pa.types.is_int64(arr_type):
        pass
    elif pa.types.is_float32(arr_type):
        try:
            iterable = iterable.cast(pa.float64())
        except pa.ArrowInvalid:
            raise TypeError("gilmap.map cannot cast input to float64.")
    else:
        try:
            iterable = iterable.cast(pa.int64())
        except pa.ArrowInvalid:
            raise TypeError(
                "gilmap.map currently only supports arrays of integers or floats."
            )
    return iterable, return_list


def map(
    func: Callable,
    iterable: Iterable | pa.Array,
    *,
    debug: bool = False,
) -> list | pa.Array:
    """Parallel map over a numeric iterable.

    Args:
        func: Callable taking one int/float and returning one int/float.
        iterable: An iterable of numbers, or a ``pyarrow.Array``.
        debug: If True (or env ``GILMAP_DEBUG=1``), print the chosen backend.

    Returns:
        ``list`` if an iterable was passed, ``pyarrow.Array`` if a pyarrow
        array was passed. Preserves input order.
    """
    if not callable(func):
        raise TypeError("The first argument must be a callable function.")

    decision = _router.decide(func)
    if debug or _router.debug_enabled():
        print(f"[gilmap] backend={decision.backend} reason={decision.reason}")

    arr, return_list = _to_arrow(iterable)

    # Fast paths (numba native, Arrow kernel, JIT) operate on the Arrow array
    # directly and skip the sub-interpreter pool entirely.
    if decision.fast_path is not None:
        result = decision.fast_path(func, arr)
        # Coerce result to expected output type (int64/float64) to keep
        # downstream byte-equality checks clean.
        if pa.types.is_float64(arr.type) and not pa.types.is_float64(result.type):
            result = result.cast(pa.float64())
        elif pa.types.is_int64(arr.type) and not pa.types.is_int64(result.type):
            try:
                result = result.cast(pa.int64())
            except pa.ArrowInvalid:
                # Some kernels (e.g. divide on ints) return floats — preserve
                # the kernel-native dtype rather than lossy-cast.
                pass
        if return_list:
            return result.to_pylist()
        return result

    # Sub-interpreter fallback path. Requires importable module-level fn.
    if _FREE_THREADED:
        raise RuntimeError(
            "gilmap.map fell back to the sub-interpreter pool on a free-threaded "
            "Python build. PEP 684 sub-interpreters with own-GIL are unavailable "
            "when Py_GIL_DISABLED is set. Either rewrite the function in a form "
            "the router can lower (Arrow kernel / numba @njit), or run on a "
            "standard GIL-enabled CPython build."
        )

    module_name, func_name = _validate_for_subinterp(func)
    result_array = execute(module_name, func_name, arr, sys.path)
    if result_array is not None:
        if return_list:
            return result_array.to_pylist()
        return result_array
    return [] if return_list else pa.array([])
