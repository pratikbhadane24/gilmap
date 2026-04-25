"""Public Python API for hyperfunctions.

This module exposes ``hyperfunctions.map``, a parallel map-style helper backed by
Rust worker threads and Python sub-interpreters.
"""

from collections.abc import Callable, Iterable
import atexit
import sys

import pyarrow as pa

from _hyperfunctions import execute, shutdown_workers

atexit.register(shutdown_workers)

def map(func: Callable[[int], int], iterable: Iterable[int] | pa.Array) -> list[int] | pa.Array:
    """Execute a module-level Python callable over integer data in parallel.

    The function must be importable by module/name from worker sub-interpreters.
    To enforce this, lambdas, local functions, and functions defined in
    ``__main__`` are rejected.

    Args:
        func: Callable that accepts one integer and returns one integer.
        iterable: Iterable of integers or a PyArrow array castable to ``int64``.

    Returns:
        A ``list[int]`` if an iterable was passed, or a ``pyarrow.Array`` if a PyArrow array was passed, preserving input order.

    Raises:
        TypeError: If ``func`` is not callable or input is not integer-castable.
        ValueError: If ``func`` is a lambda/local function or defined in ``__main__``.
        RuntimeError: If worker execution fails in Rust/Python workers.
    """
    if not callable(func):
        raise TypeError("The first argument must be a callable function.")
        
    func_name = getattr(func, "__name__", "")
    if func_name == "<lambda>":
        raise ValueError("hyperfunctions.map does not support lambda functions. Please pass a module-level function.")
    
    if "<locals>" in getattr(func, "__qualname__", ""):
        raise ValueError("hyperfunctions.map does not support local functions. Please pass a module-level function.")

    module_name = getattr(func, "__module__", "")
    if module_name == "__main__":
        raise ValueError(
            "hyperfunctions.map cannot execute functions defined in the __main__ script directly. "
            "Please define your function in a separate module and import it."
        )

    # Convert the iterable to a PyArrow array for zero-copy(ish) passage to Rust
    return_list = False
    if not isinstance(iterable, pa.Array):
        iterable = pa.array(iterable)
        return_list = True
        
    if not pa.types.is_int64(iterable.type):
        try:
            # Attempt to cast to int64 if it's not already
            iterable = iterable.cast(pa.int64())
        except pa.ArrowInvalid:
            raise TypeError("hyperfunctions.map currently only supports arrays of integers.")

    # Call the Rust execution engine
    # execute() will spawn sub-interpreters and run the function
    # It returns a PyArrow array of the results
    result_array = execute(module_name, func_name, iterable, sys.path)
    
    # We convert back to list for dead-simple "plain python" compatibility if a list was passed
    if result_array is not None:
        if return_list:
            return result_array.to_pylist()
        return result_array
        
    return [] if return_list else pa.array([], type=pa.int64())
