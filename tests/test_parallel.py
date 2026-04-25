import time
import hyperfunctions
import sys
import os

# Ensure the tests directory is in sys.path so 'tasks' can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tasks import slow_square

def test_hyperfunctions():
    args = list(range(10))
    
    start = time.time()
    result = hyperfunctions.map(slow_square, args)
    end = time.time()
    
    print(f"Hyperfunctions result: {result}")
    print(f"Hyperfunctions time: {end - start:.2f}s")
    
    # Compare with standard map
    start2 = time.time()
    expected = list(map(slow_square, args))
    end2 = time.time()
    
    print(f"Standard map result: {expected}")
    print(f"Standard map time: {end2 - start2:.2f}s")
    
    assert result == expected
    # In a truly parallel scenario, hyperfunctions time should be less than standard map time
    # on a multi-core machine.
    if end - start < end2 - start2:
        print("Parallelism achieved!")
    else:
        print("Parallelism not obvious (might be due to small input or overhead)")

if __name__ == "__main__":
    test_hyperfunctions()
