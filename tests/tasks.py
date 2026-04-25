def slow_square(x):
    # A CPU-bound task that blocks the thread
    count = 0
    for _ in range(1_000_000):
        count += 1
    return x * x

def count_primes(n):
    # Counts primes up to n, a good CPU heavy task
    if n < 2:
        return 0
    count = 0
    for i in range(2, n + 1):
        is_prime = True
        for j in range(2, int(i**0.5) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    return count

def heavy_collatz(n):
    # Runs collatz sequence many times to simulate heavy workload
    steps_total = 0
    # Run it 1000 times for each input to amplify the load
    for _ in range(1000):
        num = n
        steps = 0
        while num > 1:
            if num % 2 == 0:
                num = num // 2
            else:
                num = 3 * num + 1
            steps += 1
        steps_total += steps
    return steps_total

def quick_collatz(n: int) -> int:
    if n % 2 == 0:
        return n // 2
    return 3 * n + 1

def float_math(x: float) -> float:
    return x * 1.5 + 0.5

