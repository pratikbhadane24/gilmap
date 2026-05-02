import time
import gilmap
import sys
import os

# Ensure the tests directory is in sys.path so 'tasks' can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tasks import slow_square

def test_gilmap():
    args = list(range(10))
    
    start = time.time()
    result = gilmap.map(slow_square, args)
    end = time.time()
    
    print(f"gilmap result: {result}")
    print(f"gilmap time: {end - start:.2f}s")
    
    # Compare with standard map
    start2 = time.time()
    expected = list(map(slow_square, args))
    end2 = time.time()
    
    print(f"Standard map result: {expected}")
    print(f"Standard map time: {end2 - start2:.2f}s")
    
    assert result == expected
    # In a truly parallel scenario, gilmap time should be less than standard map time
    # on a multi-core machine.
    assert end - start < end2 - start2, "Parallelism not achieved!"
    print("Parallelism achieved!")

if __name__ == "__main__":
    test_gilmap()
