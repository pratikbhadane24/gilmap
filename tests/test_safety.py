import pytest
import hyperfunctions
import tasks
import sys
import time

def fake_main_func(x): 
    return x
fake_main_func.__module__ = "__main__"

def faulty_func(x):
    raise ValueError("Something went terribly wrong")

def fail_fast_then_sleep(x):
    if x == 0:
        raise ValueError("Early failure for chunk 0")
    time.sleep(0.01)
    return x

sys.modules['fake_module'] = type('Fake', (), {
    'faulty_func': faulty_func,
    'fail_fast_then_sleep': fail_fast_then_sleep,
})
faulty_func.__module__ = 'fake_module'
fail_fast_then_sleep.__module__ = 'fake_module'

def test_lambda_rejection():
    with pytest.raises(ValueError, match="does not support lambda functions"):
        hyperfunctions.map(lambda x: x*2, [1, 2, 3])

def test_local_function_rejection():
    def local_func(x):
        return x * 2

    with pytest.raises(ValueError, match="does not support local functions"):
        hyperfunctions.map(local_func, [1, 2, 3])

def test_main_module_rejection():
    with pytest.raises(ValueError, match="cannot execute functions defined in the __main__ script"):
        hyperfunctions.map(fake_main_func, [1, 2, 3])

def test_invalid_array_type():
    with pytest.raises(TypeError, match="currently only supports arrays of integers"):
        hyperfunctions.map(tasks.slow_square, ["a", "b", "c"])

def test_exception_in_worker_propagated():
    with pytest.raises(RuntimeError, match="Python error in worker thread"):
        hyperfunctions.map(faulty_func, [1, 2, 3])

def test_error_with_inflight_chunks_does_not_corrupt_subsequent_calls():
    for _ in range(3):
        with pytest.raises(RuntimeError, match="Python error in worker thread"):
            hyperfunctions.map(fail_fast_then_sleep, list(range(256)))

    assert hyperfunctions.map(tasks.quick_collatz, [1, 2, 3, 4]) == [4, 1, 10, 2]
