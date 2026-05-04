"""Numba dispatch tests. Skipped if numba isn't installed.

Goal: prove gilmap.map detects @njit functions, routes to the numba_native
backend, and produces results identical to calling the function directly.
"""

import importlib.util

import pytest

import gilmap


numba_spec = importlib.util.find_spec("numba")
pytestmark = pytest.mark.skipif(numba_spec is None, reason="numba not installed")


def _njit_arrow_shape():
    """Body lowers cleanly to pyarrow.compute — router picks arrow_kernel."""
    from numba import njit

    @njit(cache=False)
    def square_plus_one(x):
        return x * x + 1

    return square_plus_one


def _njit_loop_body():
    """Body has a counted loop — router can't lower to Arrow, so numba_native
    is picked over the sub-interp fallback."""
    from numba import njit

    @njit(cache=False)
    def sum_first(n):
        s = 0
        for i in range(n):
            s += i
        return s

    return sum_first


def test_arrow_kernel_wins_over_numba_for_lowerable_body():
    # Even with @njit, an arrow-kernel-shape body should route to
    # arrow_kernel because pyarrow.compute beats numba's scalar dispatcher.
    f = _njit_arrow_shape()
    d = gilmap.explain(f)
    assert d["backend"] == "arrow_kernel"


def test_numba_native_for_loop_body():
    f = _njit_loop_body()
    f(0)  # warmup compile
    d = gilmap.explain(f)
    assert d["backend"] == "numba_native"
    assert d["has_fast_path"] is True


def test_njit_results_match_python_arrow_shape():
    f = _njit_arrow_shape()
    data = [1, 2, 3, 4, 5]
    f(0)  # warmup
    out = gilmap.map(f, data)
    assert out == [x * x + 1 for x in data]


def test_njit_results_match_python_floats_arrow_shape():
    from numba import njit

    @njit(cache=False)
    def fma(x):
        return x * 1.5 + 0.5

    fma(0.0)
    data = [float(i) for i in range(20)]
    out = gilmap.map(fma, data)
    expected = [x * 1.5 + 0.5 for x in data]
    for a, b in zip(out, expected):
        assert abs(a - b) < 1e-9


def test_njit_loop_results_match_python():
    f = _njit_loop_body()
    f(0)  # warmup
    data = [3, 5, 8, 12, 20]
    out = gilmap.map(f, data)
    expected = [sum(range(n)) for n in data]
    assert out == expected


# ---------------------------------------------------------------------------
# @cfunc raw-pointer dispatch (Move C)
# ---------------------------------------------------------------------------


def _cfunc_f64():
    """Body has a counted loop — arrow_kernel rejects (not a single Return),
    so numba_cfunc gets to fire (and beats both numba_native and JIT
    because it's first in the detector list and skips the numpy round-trip).
    """
    from numba import cfunc, types

    @cfunc(types.float64(types.float64))
    def acc_loop(x):
        s = 0.0
        for i in range(10):
            s += x * float(i)
        return s

    return acc_loop


def _cfunc_i64():
    """Same shape, integer flavor — body uses `%` which arrow_kernel rejects
    (Python's `%` semantics for negatives don't match pyarrow.compute)."""
    from numba import cfunc, types

    @cfunc(types.int64(types.int64))
    def loop_mod(n):
        s = 0
        for i in range(5):
            s += (n + i) % 7
        return s

    return loop_mod


def test_cfunc_routes_to_raw_pointer_backend():
    f = _cfunc_f64()
    d = gilmap.explain(f)
    assert d["backend"] == "numba_cfunc"
    assert d["has_fast_path"] is True


def test_cfunc_f64_results_match_python():
    f = _cfunc_f64()
    data = [float(i) for i in range(50)]
    out = gilmap.map(f, data)
    expected = [sum(x * i for i in range(10)) for x in data]
    for a, b in zip(out, expected):
        assert abs(a - b) < 1e-9


def test_cfunc_i64_results_match_python():
    f = _cfunc_i64()
    data = [-12, -5, 0, 1, 7, 42, 100]
    out = gilmap.map(f, data)
    expected = [sum((n + i) % 7 for i in range(5)) for n in data]
    assert out == expected


def test_cfunc_input_dtype_cast_when_mismatched():
    """Caller passed an int list, cfunc declared float64. Router casts."""
    f = _cfunc_f64()
    data = [1, 2, 3, 4]  # int input
    out = gilmap.map(f, data)
    expected = [sum(float(x) * i for i in range(10)) for x in data]
    for a, b in zip(out, expected):
        assert abs(a - b) < 1e-9


def test_cfunc_unsupported_signature_falls_through_to_numpy_path():
    """A cfunc whose return dtype differs from arg dtype isn't handled by
    cfunc_apply Phase 1 — must fall back to the legacy numba_native path
    (which goes through the numpy round-trip and still produces correct
    results).
    """
    from numba import cfunc, types

    @cfunc(types.int64(types.float64))
    def truncate(x):
        return int(x)

    d = gilmap.explain(truncate)
    assert d["backend"] == "numba_native"


def test_cfunc_parity_with_njit():
    """Same body under @cfunc (raw pointer) and @njit (numpy round-trip)
    must produce identical results. Body has a counted loop so neither
    routes to arrow_kernel.
    """
    from numba import cfunc, njit, types

    @cfunc(types.int64(types.int64))
    def cfunc_kernel(n):
        s = 0
        for i in range(8):
            s += (n * i) - (n + i)
        return s

    @njit(cache=False)
    def njit_kernel(n):
        s = 0
        for i in range(8):
            s += (n * i) - (n + i)
        return s

    njit_kernel(0)  # warmup
    data = [-10, -1, 0, 1, 5, 17]
    cfunc_out = gilmap.map(cfunc_kernel, data)
    njit_out = gilmap.map(njit_kernel, data)
    assert cfunc_out == njit_out


def test_cfunc_empty_array():
    f = _cfunc_f64()
    out = gilmap.map(f, [])
    assert out == []
