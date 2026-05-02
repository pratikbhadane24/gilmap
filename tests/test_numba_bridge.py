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
