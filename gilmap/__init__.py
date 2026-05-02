"""Public Python API for gilmap.

This module exposes ``gilmap.map``, a parallel map-style helper backed by
Rust worker threads and Python sub-interpreters.
"""

from collections.abc import Callable, Iterable
import atexit
import sys
import sysconfig

import pyarrow as pa

from _gilmap import execute, shutdown_workers

# gilmap relies on Py_NewInterpreterFromConfig with gil = OWN_GIL (PEP 684).
# Free-threaded CPython builds (PEP 703) share a single interpreter and the
# combination is not currently defined; refuse to load there with a clear error.
if sysconfig.get_config_var("Py_GIL_DISABLED"):
    raise RuntimeError(
        "gilmap does not support free-threaded CPython builds (cp313t and similar). "
        "Per-worker sub-interpreters with their own GIL are incompatible with PEP 703. "
        "Run gilmap on a standard GIL-enabled CPython 3.12 or 3.13 build."
    )

atexit.register(shutdown_workers)

def map(func: Callable, iterable: Iterable | pa.Array) -> list | pa.Array:
    """Execute a module-level Python callable over integer/float data in parallel.

    The function must be importable by module/name from worker sub-interpreters.
    To enforce this, lambdas, local functions, and functions defined in
    ``__main__`` are rejected.

    Args:
        func: Callable that accepts one integer/float and returns one integer/float.
        iterable: Iterable of integers/floats or a PyArrow array castable to ``int64`` or ``float64``.

    Returns:
        A ``list`` if an iterable was passed, or a ``pyarrow.Array`` if a PyArrow array was passed, preserving input order.

    Raises:
        TypeError: If ``func`` is not callable or input is not castable.
        ValueError: If ``func`` is a lambda/local function or defined in ``__main__``.
        RuntimeError: If worker execution fails in Rust/Python workers.
    """
    if not callable(func):
        raise TypeError("The first argument must be a callable function.")
        
    func_name = getattr(func, "__name__", "")
    if func_name == "<lambda>":
        raise ValueError("gilmap.map does not support lambda functions. Please pass a module-level function.")
    
    if "<locals>" in getattr(func, "__qualname__", ""):
        raise ValueError("gilmap.map does not support local functions. Please pass a module-level function.")

    module_name = getattr(func, "__module__", "")
    if module_name == "__main__":
        raise ValueError(
            "gilmap.map cannot execute functions defined in the __main__ script directly. "
            "Please define your function in a separate module and import it."
        )

    # Convert the iterable to a PyArrow array for zero-copy(ish) passage to Rust
    return_list = False
    if not isinstance(iterable, pa.Array):
        iterable = pa.array(iterable)
        return_list = True
        
    if pa.types.is_float64(iterable.type) or pa.types.is_float32(iterable.type):
        try:
            iterable = iterable.cast(pa.float64())
        except pa.ArrowInvalid:
            raise TypeError("gilmap.map cannot cast input to float64.")
    else:
        try:
            # Attempt to cast to int64 if it's not already
            iterable = iterable.cast(pa.int64())
        except pa.ArrowInvalid:
            raise TypeError("gilmap.map currently only supports arrays of integers or floats.")

    # Call the Rust execution engine
    # execute() will spawn sub-interpreters and run the function
    # It returns a PyArrow array of the results
    result_array = execute(module_name, func_name, iterable, sys.path)
    
    # We convert back to list for dead-simple "plain python" compatibility if a list was passed
    if result_array is not None:
        if return_list:
            return result_array.to_pylist()
        return result_array
        
    return [] if return_list else pa.array([])
