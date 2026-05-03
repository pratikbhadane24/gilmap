import pytest
import gilmap
import tasks
import sys
import time

def fake_main_func(x):
    return x
fake_main_func.__module__ = "__main__"

def fake_main_unroutable(x):
    # Comprehension is outside JIT v2 whitelist — forces subinterp path
    # where __main__ rejection then fires.
    return sum(i * x for i in range(int(x) + 1))

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

def test_lambda_with_routable_body_now_works():
    # Post-router: a lambda whose body is Arrow-kernel-shape is accepted.
    # The router lowers it to pyarrow.compute and never reaches the
    # sub-interpreter pool, so the importability requirement does not apply.
    out = gilmap.map(lambda x: x * 2, [1, 2, 3])
    assert out == [2, 4, 6]

def test_lambda_with_unroutable_body_rejected():
    # Lambdas whose bodies the router can't lower must still be rejected
    # because they would fall through to the sub-interpreter path.
    with pytest.raises(ValueError, match="does not support lambda"):
        gilmap.map(lambda x: sum(range(int(x))), [1, 2, 3])

def test_local_function_with_routable_body_now_works():
    # Same logic for nested/local functions.
    def local_func(x):
        return x * 2
    assert gilmap.map(local_func, [1, 2, 3]) == [2, 4, 6]

def test_local_function_with_unroutable_body_rejected():
    # Use a comprehension — outside JIT v2 whitelist — so router falls
    # through to subinterp where the local-fn rejection fires.
    def local_acc(x):
        return sum(i * x for i in range(int(x) + 1))
    with pytest.raises(ValueError, match="local"):
        gilmap.map(local_acc, [1, 2, 3])

def test_main_module_function_with_routable_body_now_works():
    # __main__-defined fns get the same router treatment when the body lowers.
    assert gilmap.map(fake_main_func, [1, 2, 3]) == [1, 2, 3]

def test_main_module_function_with_unroutable_body_rejected():
    # Promoted to module level (no <locals> in qualname) so the check the
    # test asserts (__main__ module rejection) is the one that fires.
    fake_main_unroutable.__module__ = "__main__"
    with pytest.raises(ValueError, match="__main__"):
        gilmap.map(fake_main_unroutable, [1, 2, 3])

def test_invalid_array_type():
    with pytest.raises(TypeError, match="currently only supports arrays of integers"):
        gilmap.map(tasks.slow_square, ["a", "b", "c"])

def test_exception_in_worker_propagated():
    with pytest.raises(RuntimeError, match="Python error in worker thread"):
        gilmap.map(faulty_func, [1, 2, 3])

def test_error_with_inflight_chunks_does_not_corrupt_subsequent_calls():
    for _ in range(3):
        with pytest.raises(RuntimeError, match="Python error in worker thread"):
            gilmap.map(fail_fast_then_sleep, list(range(256)))

    assert gilmap.map(tasks.quick_collatz, [1, 2, 3, 4]) == [4, 1, 10, 2]
