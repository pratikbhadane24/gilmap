"""Free-threaded build (PEP 703 / cp313t / cp314t) gating.

These tests guard the contract that:
  - On a GIL-enabled build, gilmap loads normally and the sub-interp path is
    available.
  - On a free-threaded build, gilmap still loads (we stopped raising at
    import in the P1+ rewrite); router fast paths still work; the
    sub-interpreter fallback raises a targeted RuntimeError so callers know
    to rewrite for an Arrow-kernel / numba shape.
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
def test_subinterp_fallback_raises_clear_error_on_free_threaded():
    """On a 3.13t/3.14t build, hitting the sub-interp path must produce a
    clear, actionable error pointing the user at the router's fast paths.
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from tasks import slow_square  # body has a for-loop, not router-routable

    with pytest.raises(RuntimeError, match="free-threaded"):
        gilmap.map(slow_square, list(range(5)))
