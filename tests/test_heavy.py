import time
import multiprocessing
import hyperfunctions
import sys
import os

# Ensure the tests directory is in sys.path so 'tasks' can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tasks import count_primes, heavy_collatz, quick_collatz

def run_benchmark(name, func, args):
    print(f"--- Benchmark: {name} ---")
    
    # 1. Standard python map (single-threaded)
    start_std = time.time()
    expected = list(map(func, args))
    end_std = time.time()
    print(f"Standard map: {end_std - start_std:.4f}s")
    
    # 2. multiprocessing.Pool
    # Use context manager to properly clean up processes
    start_mp = time.time()
    with multiprocessing.Pool() as p:
        res_mp = p.map(func, args)
    end_mp = time.time()
    print(f"Multiprocessing map: {end_mp - start_mp:.4f}s")
    
    assert res_mp == expected, "Multiprocessing returned wrong results"
    
    # 3. hyperfunctions.map
    start_hf = time.time()
    res_hf = hyperfunctions.map(func, args)
    end_hf = time.time()
    print(f"Hyperfunctions map: {end_hf - start_hf:.4f}s")
    
    assert res_hf == expected, "Hyperfunctions returned wrong results"
    
    print(f"Speedup vs Standard map: {(end_std - start_std) / (end_hf - start_hf):.2f}x")
    print(f"Speedup vs Multiprocessing: {(end_mp - start_mp) / (end_hf - start_hf):.2f}x\n")

if __name__ == "__main__":
    print("Beginning Battle Tests...\n")
    
    # Test 1: Count Primes
    # The workload is highly variable depending on the number (larger numbers take longer)
    # We will use large numbers to ensure the workload is substantial.
    primes_args = [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000]
    run_benchmark("Count Primes", count_primes, primes_args)
    
    # Test 2: Heavy Collatz
    # The workload is relatively unpredictable but heavy iterative math
    # We use a large range of inputs
    collatz_args = list(range(1000, 2000))
    run_benchmark("Heavy Collatz (1000 numbers)", heavy_collatz, collatz_args)
    
    # Test 3: Large Dataset Overhead Test
    # This will test the overhead of passing a huge number of arguments back and forth
    print("--- Overhead test with minimal computation ---")
    overhead_args = list(range(1, 1_000_000))
    
    # run standard map
    s = time.time()
    _ = list(map(quick_collatz, overhead_args))
    print(f"Standard map (overhead): {time.time() - s:.4f}s")
    
    # run multiprocessing map
    s = time.time()
    with multiprocessing.Pool() as p:
        _ = p.map(quick_collatz, overhead_args)
    print(f"Multiprocessing (overhead): {time.time() - s:.4f}s")
    
    # run hyperfunctions
    s = time.time()
    _ = hyperfunctions.map(quick_collatz, overhead_args)
    print(f"Hyperfunctions (overhead, list): {time.time() - s:.4f}s")

    # run hyperfunctions with arrow
    import pyarrow as pa
    arrow_args = pa.array(overhead_args)
    s = time.time()
    _ = hyperfunctions.map(quick_collatz, arrow_args)
    print(f"Hyperfunctions (overhead, arrow): {time.time() - s:.4f}s")
