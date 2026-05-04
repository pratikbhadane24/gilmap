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


# ---------------------------------------------------------------------------
# SIMD-vectorized JIT (Phase 1: 128-bit lanes, single-Return pure-expr bodies)
# ---------------------------------------------------------------------------


def test_simd_f64_arithmetic():
    """Pure-expression f64 body → vector main loop (F64X2)."""
    def fma(x):
        return x * 2.0 + 1.0

    # 1000 elements: 500 vector iterations, no scalar tail.
    data = [float(i) for i in range(1000)]
    out = _apply(fma, "f64", data)
    expected = [fma(x) for x in data]
    for a, b in zip(out, expected):
        assert abs(a - b) < 1e-12


def test_simd_i64_arithmetic():
    """Pure-expression i64 body → vector main loop (I64X2)."""
    def f(x):
        return x * 3 - 5

    data = list(range(-50, 51))  # 101 elements: 50 vector pairs + 1 tail
    out = _apply(f, "i64", data)
    expected = [f(x) for x in data]
    assert out == expected


def test_simd_scalar_tail_runs():
    """len % SIMD_LANES != 0 forces the scalar-tail loop. With LANES=2,
    any odd-length input exercises the tail path."""
    def f(x):
        return x + 7

    # 13 elements → 6 vector iterations + 1 scalar tail.
    data = list(range(13))
    out = _apply(f, "i64", data)
    assert out == [x + 7 for x in data]


def test_simd_compare_select():
    """Ternary in vectorizable body lowers to vector mask + bitselect."""
    def clamp(x):
        return 0.0 if x < 0.0 else x

    data = [-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0]  # 7 elements: vec + tail
    out = _apply(clamp, "f64", data)
    expected = [clamp(x) for x in data]
    for a, b in zip(out, expected):
        assert abs(a - b) < 1e-12


def test_simd_unary_neg():
    def f(x):
        return -x

    data = [float(i) for i in range(-10, 11)]
    out = _apply(f, "f64", data)
    expected = [-x for x in data]
    for a, b in zip(out, expected):
        assert abs(a - b) < 1e-12


def test_simd_mixed_dtype_body_falls_back_to_scalar_path():
    """math.sqrt forces non-vectorizable (MathCall in body). The scalar
    codegen path must still produce correct results — no regression.
    """
    def f(x):
        return math.sqrt(x) * 2.0

    data = [float(i) for i in range(1, 21)]
    out = _apply(f, "f64", data)
    expected = [f(x) for x in data]
    for a, b in zip(out, expected):
        assert abs(a - b) < 1e-12


def test_simd_locals_falls_back_to_scalar_path():
    """Multi-statement body with a local → scalar codegen path. Same
    answer as Python."""
    def f(x):
        s = x + 1
        return s * s

    data = list(range(15))
    out = _apply(f, "i64", data)
    expected = [(x + 1) * (x + 1) for x in data]
    assert out == expected


def test_simd_loop_body_falls_back_to_scalar_path():
    """Counted loop in body → scalar codegen path."""
    def f(x):
        s = 0
        for _i in range(5):
            s += x
        return s

    data = list(range(10))
    out = _apply(f, "i64", data)
    expected = [x * 5 for x in data]
    assert out == expected


def test_simd_empty_input():
    def f(x):
        return x + 1

    out = _apply(f, "i64", [])
    assert out == []


def test_simd_single_element_only_tail():
    """len < SIMD_LANES → all elements go through the scalar tail."""
    def f(x):
        return x * 2

    out = _apply(f, "i64", [42])
    assert out == [84]
