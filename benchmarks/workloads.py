"""Module-level workload functions used by every comparator runner.

Functions live at module scope so per-worker Python sub-interpreters
(used by gilmap) can re-import them by `(module, name)`. Lambdas and
nested functions would not be importable across sub-interpreters.

Each function takes a single int or float and returns one int or float
to match the gilmap.map contract. NumPy/numba variants for vectorizable
workloads live in `runners.py` so this module stays SDK-clean.
"""


def count_primes(n: int) -> int:
    """Count primes up to n via trial division. Heavy int compute."""
    if n < 2:
        return 0
    count = 0
    for i in range(2, n + 1):
        is_prime = True
        j = 2
        while j * j <= i:
            if i % j == 0:
                is_prime = False
                break
            j += 1
        if is_prime:
            count += 1
    return count


def heavy_collatz(n: int) -> int:
    """Iterate Collatz 1000x to amplify per-element load."""
    steps_total = 0
    for _ in range(1000):
        num = n if n > 0 else 1
        steps = 0
        while num > 1:
            num = num // 2 if num % 2 == 0 else 3 * num + 1
            steps += 1
        steps_total += steps
    return steps_total


def mandelbrot_iters(x: float) -> int:
    """Mandelbrot escape-time at c = (x/1000) + (x/1500)i. Heavy float compute."""
    cr = x / 1000.0 - 0.5
    ci = x / 1500.0
    zr = 0.0
    zi = 0.0
    max_iter = 800
    for i in range(max_iter):
        zr2 = zr * zr
        zi2 = zi * zi
        if zr2 + zi2 > 4.0:
            return i
        zi = 2.0 * zr * zi + ci
        zr = zr2 - zi2 + cr
    return max_iter


def medium_compute(n: int) -> int:
    """~10us of pure-Python work. Crossover-zone workload."""
    acc = 0
    for i in range(200):
        acc = (acc + i * n) % 1_000_003
    return acc


def quick_collatz(n: int) -> int:
    """One Collatz step. Trivial; overhead-dominated benchmark."""
    return n // 2 if n % 2 == 0 else 3 * n + 1


def float_math(x: float) -> float:
    """Trivial float op. NumPy/numba should beat parallel approaches here."""
    return x * 1.5 + 0.5
