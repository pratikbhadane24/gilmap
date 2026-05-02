"""Comparator runners. Uniform interface, optional deps skipped cleanly.

Each runner exposes:
    name: str
    available() -> tuple[bool, str]    # (ok, skip_reason)
    supports(workload, container) -> bool
    setup() -> None                    # one-time process setup (e.g. ray.init)
    run(func, data) -> Any             # actual call to time

Per-runner setup happens once at process start; the timed `run()` should
contain only the parallel-execution call so steady-state numbers exclude
pool initialization. gilmap's first-call cost is captured as `warmup_s`.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable

from . import workloads


class _Runner:
    name: str = ""
    requires_arrow: bool = False
    is_baseline: bool = False
    only_workloads: set[str] | None = None  # None = all

    def available(self) -> tuple[bool, str]:
        return True, ""

    def supports(self, workload: str, container: str) -> bool:
        if self.requires_arrow and container != "arrow":
            return False
        if (not self.requires_arrow) and container == "arrow":
            return False
        if self.only_workloads is not None and workload not in self.only_workloads:
            return False
        return True

    def setup(self) -> None:
        return

    def teardown(self) -> None:
        return

    def run(self, func: Callable, data: Any) -> Any:
        raise NotImplementedError


# ---- baseline -----------------------------------------------------------

class StdMap(_Runner):
    name = "std_map"
    is_baseline = True

    def run(self, func: Callable, data: Any) -> Any:
        return list(map(func, data))


# ---- stdlib parallel ----------------------------------------------------

class MpPool(_Runner):
    name = "mp_pool"
    _pool: mp.pool.Pool | None = None

    def setup(self) -> None:
        self._pool = mp.Pool()

    def teardown(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def run(self, func: Callable, data: Any) -> Any:
        assert self._pool is not None
        return self._pool.map(func, list(data))


def _fair_chunksize(n: int) -> int:
    import os
    cores = os.cpu_count() or 4
    return max(1, n // (cores * 4))


class CfProcess(_Runner):
    name = "cf_process"
    _ex: ProcessPoolExecutor | None = None

    def setup(self) -> None:
        self._ex = ProcessPoolExecutor()

    def teardown(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True)
            self._ex = None

    def run(self, func: Callable, data: Any) -> Any:
        assert self._ex is not None
        items = list(data)
        return list(self._ex.map(func, items, chunksize=_fair_chunksize(len(items))))


class CfThread(_Runner):
    name = "cf_thread"
    _ex: ThreadPoolExecutor | None = None

    def setup(self) -> None:
        # GIL-bound; included to demonstrate why naive threading fails for CPU work.
        import os

        self._ex = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

    def teardown(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True)
            self._ex = None

    def run(self, func: Callable, data: Any) -> Any:
        assert self._ex is not None
        items = list(data)
        return list(self._ex.map(func, items, chunksize=_fair_chunksize(len(items))))


# ---- gilmap -------------------------------------------------------------

class GilmapList(_Runner):
    name = "gilmap_list"

    def run(self, func: Callable, data: Any) -> Any:
        import gilmap

        return gilmap.map(func, list(data))


class GilmapArrow(_Runner):
    name = "gilmap_arrow"
    requires_arrow = True

    def run(self, func: Callable, data: Any) -> Any:
        import gilmap

        # data is already a pyarrow.Array supplied by the harness
        return gilmap.map(func, data)


# ---- joblib -------------------------------------------------------------

class Joblib(_Runner):
    name = "joblib"
    _parallel = None

    def available(self) -> tuple[bool, str]:
        try:
            import joblib  # noqa: F401
        except ImportError:
            return False, "joblib not installed"
        return True, ""

    def setup(self) -> None:
        from joblib import Parallel

        # loky backend = process pool; n_jobs=-1 uses all cores
        self._parallel = Parallel(n_jobs=-1, backend="loky", return_as="list")

    def run(self, func: Callable, data: Any) -> Any:
        from joblib import delayed

        assert self._parallel is not None
        return self._parallel(delayed(func)(x) for x in data)


# ---- numpy vector -------------------------------------------------------

class NumpyVec(_Runner):
    """Pure NumPy vectorized form. Only meaningful for vectorizable workloads."""

    name = "numpy_vec"
    only_workloads = {"float_math", "quick_collatz"}

    def available(self) -> tuple[bool, str]:
        try:
            import numpy  # noqa: F401
        except ImportError:
            return False, "numpy not installed"
        return True, ""

    def run(self, func: Callable, data: Any) -> Any:
        import numpy as np

        arr = np.asarray(list(data))
        name = getattr(func, "__name__", "")
        if name == "float_math":
            return (arr * 1.5 + 0.5).tolist()
        if name == "quick_collatz":
            even = arr % 2 == 0
            out = np.where(even, arr // 2, 3 * arr + 1)
            return out.tolist()
        raise NotImplementedError(f"numpy_vec has no form for {name}")


# ---- numba --------------------------------------------------------------

class Numba(_Runner):
    """Numba @njit(parallel=True) form. Only for workloads with a JIT-friendly form."""

    name = "numba"
    only_workloads = {"float_math", "quick_collatz", "mandelbrot_iters"}
    _compiled: dict[str, Callable] = {}

    def available(self) -> tuple[bool, str]:
        try:
            import numba  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            return False, "numba/numpy not installed"
        return True, ""

    def setup(self) -> None:
        import numpy as np
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _float_math(arr):
            out = np.empty_like(arr)
            for i in prange(arr.shape[0]):
                out[i] = arr[i] * 1.5 + 0.5
            return out

        @njit(parallel=True, cache=True)
        def _quick_collatz(arr):
            out = np.empty_like(arr)
            for i in prange(arr.shape[0]):
                v = arr[i]
                if v % 2 == 0:
                    out[i] = v // 2
                else:
                    out[i] = 3 * v + 1
            return out

        @njit(parallel=True, cache=True)
        def _mandelbrot(arr):
            out = np.empty(arr.shape[0], dtype=np.int64)
            for i in prange(arr.shape[0]):
                x = arr[i]
                cr = x / 1000.0 - 0.5
                ci = x / 1500.0
                zr = 0.0
                zi = 0.0
                max_iter = 800
                k = max_iter
                for j in range(max_iter):
                    zr2 = zr * zr
                    zi2 = zi * zi
                    if zr2 + zi2 > 4.0:
                        k = j
                        break
                    zi = 2.0 * zr * zi + ci
                    zr = zr2 - zi2 + cr
                out[i] = k
            return out

        # Force compilation now (counts as setup, not steady state).
        _float_math(np.asarray([0.0, 1.0]))
        _quick_collatz(np.asarray([1, 2], dtype=np.int64))
        _mandelbrot(np.asarray([0.0, 1.0]))
        self._compiled = {
            "float_math": _float_math,
            "quick_collatz": _quick_collatz,
            "mandelbrot_iters": _mandelbrot,
        }

    def run(self, func: Callable, data: Any) -> Any:
        import numpy as np

        name = getattr(func, "__name__", "")
        kernel = self._compiled[name]
        if name in ("float_math", "mandelbrot_iters"):
            arr = np.asarray(list(data), dtype=np.float64)
        else:
            arr = np.asarray(list(data), dtype=np.int64)
        return kernel(arr).tolist()


# ---- ray ----------------------------------------------------------------

class Ray(_Runner):
    name = "ray"

    def available(self) -> tuple[bool, str]:
        try:
            import ray  # noqa: F401
        except ImportError:
            return False, "ray not installed"
        return True, ""

    def setup(self) -> None:
        import ray

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                logging_level="ERROR",
                log_to_driver=False,
            )

    def teardown(self) -> None:
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass

    def run(self, func: Callable, data: Any) -> Any:
        import ray

        remote_fn = ray.remote(func)
        # Chunk to amortize remote-task overhead. One ray task per element
        # is catastrophic; this is the canonical ray idiom.
        items = list(data)
        n = len(items)
        cores = 8 if n > 0 else 1
        try:
            import os

            cores = max(1, (os.cpu_count() or 4))
        except Exception:
            pass
        chunk_size = max(1, n // (cores * 4) or 1)

        @ray.remote
        def _chunk(xs):
            return [func(x) for x in xs]

        chunks = [items[i : i + chunk_size] for i in range(0, n, chunk_size)]
        futs = [_chunk.remote(c) for c in chunks]
        out: list = []
        for r in ray.get(futs):
            out.extend(r)
        # silence unused
        _ = remote_fn
        return out


# ---- dask ---------------------------------------------------------------

class Dask(_Runner):
    name = "dask"

    def available(self) -> tuple[bool, str]:
        try:
            import dask  # noqa: F401
            import dask.bag  # noqa: F401
        except ImportError:
            return False, "dask not installed"
        return True, ""

    def run(self, func: Callable, data: Any) -> Any:
        import dask.bag as db

        items = list(data)
        # npartitions ~ cores so chunks are big enough to amortize task overhead.
        import os

        cores = os.cpu_count() or 4
        nparts = max(1, min(len(items), cores * 2))
        bag = db.from_sequence(items, npartitions=nparts)
        return bag.map(func).compute(scheduler="processes")


ALL_RUNNERS: list[type[_Runner]] = [
    StdMap,
    MpPool,
    CfProcess,
    CfThread,
    GilmapList,
    GilmapArrow,
    Joblib,
    NumpyVec,
    Numba,
    Ray,
    Dask,
]


def build_runners(names: list[str] | None = None) -> list[_Runner]:
    out: list[_Runner] = []
    for cls in ALL_RUNNERS:
        if names is not None and cls.name not in names:
            continue
        out.append(cls())
    return out


# Tiny self-check: ensure workloads module is importable and runners can find
# functions by name. Catches typos before the harness runs anything heavy.
def smoke_workloads() -> dict[str, Callable]:
    return {
        "count_primes": workloads.count_primes,
        "heavy_collatz": workloads.heavy_collatz,
        "mandelbrot_iters": workloads.mandelbrot_iters,
        "medium_compute": workloads.medium_compute,
        "quick_collatz": workloads.quick_collatz,
        "float_math": workloads.float_math,
    }
