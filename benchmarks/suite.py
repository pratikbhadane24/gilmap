"""Declarative benchmark scenarios."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    workload: str  # name in workloads.py
    lane: str  # "int64" | "float64"
    sizes: tuple[int, ...]
    quick_size: int  # used by --quick


SCENARIOS: tuple[Scenario, ...] = (
    # Heavy compute — gilmap should shine
    Scenario("count_primes", "int64", sizes=(32, 128, 512), quick_size=8),
    Scenario("heavy_collatz", "int64", sizes=(64, 256, 512), quick_size=64),
    Scenario("mandelbrot_iters", "float64", sizes=(1_000, 10_000, 25_000), quick_size=1_000),
    # Crossover zone
    Scenario("medium_compute", "int64", sizes=(1_000, 10_000, 100_000), quick_size=1_000),
    # Overhead-bound — gilmap may lose
    Scenario("quick_collatz", "int64", sizes=(10_000, 100_000, 1_000_000), quick_size=10_000),
    # Vectorizable — numpy/numba should beat gilmap honestly
    Scenario("float_math", "float64", sizes=(10_000, 100_000, 1_000_000), quick_size=10_000),
)


def workload_lane(name: str) -> str:
    for s in SCENARIOS:
        if s.workload == name:
            return s.lane
    raise KeyError(name)


def make_inputs(workload: str, size: int, lane: str) -> list:
    """Build deterministic inputs of the right lane for `workload`."""
    if workload == "count_primes":
        # ramp up so each call is meaningful but distinct
        base = 5_000
        return [base + i * 73 for i in range(size)]
    if workload == "heavy_collatz":
        return [1000 + i for i in range(size)]
    if workload == "mandelbrot_iters":
        return [float(i) for i in range(size)]
    if workload == "medium_compute":
        return [(i * 31) % 100_003 for i in range(size)]
    if workload == "quick_collatz":
        return [i + 1 for i in range(size)]
    if workload == "float_math":
        return [float(i) for i in range(size)]
    raise KeyError(workload)
