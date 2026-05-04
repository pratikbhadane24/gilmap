"""Free-threaded build (PEP 703 / cp313t / cp314t) gating.

Contract:
  - On a GIL-enabled build, gilmap loads normally and the sub-interpreter
    fallback is available.
  - On a free-threaded build, gilmap loads, fast paths still work, and the
    fallback path runs on a rayon-based executor that shares the main
    interpreter — no longer raises.
"""

import sysconfig

import pytest

import gilmap


_FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def test_module_loads_regardless_of_gil_state():
    # If we got here, gilmap imported. That's the assertion.
    assert hasattr(gilmap, "map")
    assert hasattr(gilmap, "explain")


def test_router_fast_paths_are_advertised():
    # Trivial body must always advertise a fast path — independent of
    # whether the GIL is enabled.
    decision = gilmap.explain(lambda x: x * 2 + 1)
    assert decision["has_fast_path"] is True
    assert decision["backend"] == "arrow_kernel"


@pytest.mark.skipif(_FREE_THREADED, reason="needs GIL-enabled build")
def test_subinterp_path_works_on_gil_build():
    """On a normal CPython build, sub-interpreter fallback runs fine."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from tasks import slow_square
    out = gilmap.map(slow_square, list(range(5)))
    assert out == [x * x for x in range(5)]


@pytest.mark.skipif(not _FREE_THREADED, reason="only meaningful on free-threaded build")
def test_rayon_fallback_runs_on_free_threaded():
    """On a 3.13t/3.14t build, the fallback path must execute on the
    rayon-based shared-interpreter executor and produce correct results.
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from tasks import slow_square  # body has a for-loop, not router-routable

    out = gilmap.map(slow_square, list(range(5)))
    assert out == [x * x for x in range(5)]


@pytest.mark.skipif(not _FREE_THREADED, reason="only meaningful on free-threaded build")
def test_rayon_fallback_handles_floats():
    """f64 path of the rayon executor."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from tasks import float_math

    out = gilmap.map(float_math, [0.0, 1.0, 2.0, 3.0])
    assert out == [float_math(x) for x in [0.0, 1.0, 2.0, 3.0]]


@pytest.mark.skipif(not _FREE_THREADED, reason="only meaningful on free-threaded build")
def test_rayon_fallback_handles_empty_input():
    """Empty input is a no-op, not an error, on the rayon executor."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from tasks import slow_square

    assert gilmap.map(slow_square, []) == []
