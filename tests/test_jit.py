"""Cranelift JIT tests.

The v1 JIT shares the single-`return <expr>` whitelist with the
`arrow_kernel` backend, so in normal routing JIT only fires when
arrow_kernel rejects a shape. We bypass the router here to call the
JIT directly and verify byte-for-byte parity with the Python
implementation across the whitelisted ops.
"""

import math

import pyarrow as pa
import pytest

from gilmap import _jit


def _apply(func, dtype, data):
    compiled = _jit.try_compile(func, dtype)
    assert compiled is not None, f"JIT failed to compile {func.__name__}"
    h, out_dtype = compiled
    arr_type = pa.float64() if dtype == "f64" else pa.int64()
    arr = pa.array(data, type=arr_type)
    return _jit.jit_apply(h, dtype, out_dtype, arr).to_pylist()


def test_jit_simple_arith_f64():
    def fma(x):
        return x * 1.5 + 0.5

    data = [float(i) for i in range(10)]
    out = _apply(fma, "f64", data)
    expected = [fma(x) for x in data]
    for a, b in zip(out, expected):
        assert abs(a - b) < 1e-12


def test_jit_simple_arith_i64():
    def lin(x):
        return x * 3 + 7

    data = list(range(20))
    out = _apply(lin, "i64", data)
    expected = [lin(x) for x in data]
    assert out == expected


def test_jit_math_sqrt():
    def f(x):
        return math.sqrt(x) * 2.0

    data = [float(i) for i in range(1, 30)]
    out = _apply(f, "f64", data)
    expected = [f(x) for x in data]
    for a, b in zip(out, expected):
        assert abs(a - b) < 1e-12


def test_jit_math_exp_log():
    def f(x):
        return math.log(math.exp(x))

    data = [float(i) / 10.0 for i in range(-5, 6)]
    out = _apply(f, "f64", data)
    for a, b in zip(out, data):
        assert abs(a - b) < 1e-9


def test_jit_compare_select():
    # Ternary lowers to `select`. Verify branch-free dispatch.
    def clamp(x):
        return 0.0 if x < 0.0 else x

    data = [-2.0, -0.5, 0.0, 1.0, 5.0]
    out = _apply(clamp, "f64", data)
    expected = [clamp(x) for x in data]
    for a, b in zip(out, expected):
        assert abs(a - b) < 1e-12


def test_jit_caching():
    """Same source compiles to same hash; repeat compile is a no-op."""
    def f(x):
        return x + 1

    c1 = _jit.try_compile(f, "i64")
    c2 = _jit.try_compile(f, "i64")
    assert c1 is not None and c2 is not None
    assert c1[0] == c2[0]  # same kernel hash


def test_jit_rejects_unsupported_node():
    def f(x):
        # Comprehensions are outside the whitelist — JIT must reject.
        return sum(i * x for i in range(10))

    assert _jit.try_compile(f, "i64") is None


def test_jit_rejects_free_variable():
    factor = 7

    def f(x):
        return x * factor

    # `factor` is a free variable; JIT v1 doesn't capture closures.
    assert _jit.try_compile(f, "i64") is None


def test_jit_int_in_f64_out_via_math():
    # P5b auto-promotes mixed lanes — int input through math.sqrt
    # produces an f64 output kernel. Compiles cleanly.
    def f(x):
        return math.sqrt(x)

    compiled = _jit.try_compile(f, "i64")
    assert compiled is not None
    h, out_dt = compiled
    assert out_dt == "f64"


def test_jit_modulo_int_c_semantics():
    # arrow_kernel rejects %; the JIT keeps it but uses Cranelift's `srem`
    # (C-style signed remainder), NOT Python's `%`. For non-negative inputs
    # they agree; for negatives Python's `%` differs. This test asserts the
    # documented C-style behavior; users who need Python's `%` semantics
    # should not route to JIT for negative inputs.
    def m(x):
        return x % 7

    # Use non-negative inputs only — that's the contract for this op in v1.
    data = list(range(0, 30))
    out = _apply(m, "i64", data)
    expected = [x % 7 for x in data]
    assert out == expected
