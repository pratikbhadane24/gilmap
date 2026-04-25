import pytest
import hyperfunctions
import tasks
import sys

def fake_main_func(x): 
    return x
fake_main_func.__module__ = "__main__"

def faulty_func(x):
    raise ValueError("Something went terribly wrong")

sys.modules['fake_module'] = type('Fake', (), {'faulty_func': faulty_func})
faulty_func.__module__ = 'fake_module'

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
