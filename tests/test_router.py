"""Router-selection tests. Verify gilmap.explain picks the expected backend
and that the fast path produces the same numbers as the sub-interp fallback.
"""

import math
import os
import sys

import pyarrow as pa
import pytest

import gilmap
from gilmap import _router

sys.path.insert(0, os.path.dirname(__file__))
from tasks import slow_square, subinterp_only  # noqa: E402


def setup_function(_fn):
    _router.reset_cache()


def f_arrow_kernel(x):
    return x * 1.5 + 0.5


def f_arrow_kernel_int(x):
    return x * 3 + 7


def f_arrow_math(x):
    return math.sqrt(x) * 2.0


def f_unsupported(x):
    # while-loop is outside JIT v2 whitelist (only for-range supported);
    # falls through to sub-interpreter.
    acc = 0
    n = x
    while n > 0:
        acc += n
        n -= 1
    return acc


def test_explain_arrow_kernel_simple_arith():
    d = gilmap.explain(f_arrow_kernel)
    assert d["backend"] == "arrow_kernel"
    assert d["has_fast_path"] is True


def test_explain_arrow_kernel_math_call():
    d = gilmap.explain(f_arrow_math)
    assert d["backend"] == "arrow_kernel"


def test_explain_falls_through_for_loops():
    d = gilmap.explain(f_unsupported)
    assert d["backend"] == "subinterp"


def test_arrow_kernel_results_match_python_int():
    arr = pa.array(list(range(20)), type=pa.int64())
    out = gilmap.map(f_arrow_kernel_int, arr)
    expected = [f_arrow_kernel_int(i) for i in range(20)]
    assert out.to_pylist() == expected


def test_arrow_kernel_results_match_python_float():
    data = [float(i) for i in range(50)]
    out = gilmap.map(f_arrow_kernel, data)
    expected = [f_arrow_kernel(x) for x in data]
    for a, b in zip(out, expected):
        assert abs(a - b) < 1e-9


def test_arrow_kernel_math_sqrt():
    data = [float(i) for i in range(1, 30)]
    out = gilmap.map(f_arrow_math, data)
    expected = [f_arrow_math(x) for x in data]
    for a, b in zip(out, expected):
        assert abs(a - b) < 1e-9


def test_subinterp_fallback_still_works():
    # subinterp_only uses a comprehension — outside JIT v2 whitelist, so
    # routing falls through to sub-interp.
    d = gilmap.explain(subinterp_only)
    assert d["backend"] == "subinterp"
    out = gilmap.map(subinterp_only, [3, 4, 5])
    assert out == [subinterp_only(x) for x in [3, 4, 5]]


def test_decision_cached():
    _router.reset_cache()
    d1 = _router.decide(f_arrow_kernel)
    d2 = _router.decide(f_arrow_kernel)
    assert d1 is d2  # cache returns same object


def test_lambda_routes_to_arrow_kernel():
    """Lambdas were rejected by the old gilmap. Now they go through the
    router; if the body is Arrow-kernel-shape, it works. (Sub-interp lambdas
    still raise — verified separately.)"""
    f = lambda x: x * 2 + 1  # noqa: E731
    d = gilmap.explain(f)
    assert d["backend"] == "arrow_kernel"
    out = gilmap.map(f, [1, 2, 3, 4, 5])
    assert out == [3, 5, 7, 9, 11]


def test_lambda_with_unsupported_body_still_rejected():
    f = lambda x: sum(range(int(x)))  # noqa: E731 — body not Arrow-kernel-shape
    with pytest.raises(ValueError, match="lambda"):
        gilmap.map(f, [1, 2, 3])
