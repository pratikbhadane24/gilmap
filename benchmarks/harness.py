"""Timing core: warmup, repeat, env capture, equality check, JSON writer."""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import sysconfig
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class CellResult:
    workload: str
    size: int
    lane: str  # "int64" | "float64"
    runner: str
    container: str  # "list" | "arrow"
    n: int
    median_s: float | None
    min_s: float | None
    max_s: float | None
    stdev_s: float | None
    warmup_s: float | None
    ok: bool
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""
    unstable: bool = False  # stdev > 10% of median


@dataclass
class RunReport:
    env: dict[str, Any] = field(default_factory=dict)
    cells: list[dict[str, Any]] = field(default_factory=list)
    runner_setup_s: dict[str, float] = field(default_factory=dict)


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _git_dirty(repo_root: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return False


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "logical_cores": os.cpu_count(),
    }
    sysname = platform.system()
    try:
        if sysname == "Darwin":
            info["cpu_model"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            for oid in ("hw.physicalcpu", "hw.physcpu"):
                try:
                    info["physical_cores"] = int(
                        subprocess.check_output(
                            ["sysctl", "-n", oid],
                            text=True,
                            stderr=subprocess.DEVNULL,
                        ).strip()
                    )
                    break
                except Exception:
                    continue
        elif sysname == "Linux":
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        elif sysname == "Windows":
            info["cpu_model"] = platform.processor()
    except Exception:
        info.setdefault("cpu_model", "unknown")
    return info


def _dep_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for mod in (
        "gilmap",
        "pyarrow",
        "numpy",
        "joblib",
        "numba",
        "ray",
        "dask",
        "matplotlib",
        "psutil",
    ):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = "missing"
    return versions


def capture_env(repo_root: Path) -> dict[str, Any]:
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "build": platform.python_build(),
            "gil_disabled": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
            "executable": sys.executable,
        },
        "cpu": _cpu_info(),
        "git": {
            "commit": _git_commit(repo_root),
            "dirty": _git_dirty(repo_root),
        },
        "dep_versions": _dep_versions(),
    }


def _to_pylist(v: Any) -> list:
    """Normalize any runner output for equality comparison."""
    if hasattr(v, "to_pylist"):
        return v.to_pylist()
    if hasattr(v, "tolist"):
        return v.tolist()
    return list(v)


def _equal(a: Any, b: Any) -> bool:
    al = _to_pylist(a)
    bl = _to_pylist(b)
    if len(al) != len(bl):
        return False
    for x, y in zip(al, bl):
        if isinstance(x, float) or isinstance(y, float):
            # tolerate float noise across runners (numpy vs python math etc.)
            if abs(float(x) - float(y)) > 1e-6 * max(1.0, abs(float(x))):
                return False
        else:
            if x != y:
                return False
    return True


def time_cell(
    runner_fn: Callable[[], Any],
    *,
    n: int,
    warmup: bool,
) -> tuple[list[float], float | None, Any]:
    """Run one cell. Returns (timings, warmup_s, last_result)."""
    warmup_s: float | None = None
    if warmup:
        t0 = time.perf_counter()
        last = runner_fn()
        warmup_s = time.perf_counter() - t0
    else:
        last = None

    timings: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        last = runner_fn()
        timings.append(time.perf_counter() - t0)
    return timings, warmup_s, last


def summarize(timings: list[float]) -> dict[str, float]:
    if not timings:
        return {"median_s": None, "min_s": None, "max_s": None, "stdev_s": None}  # type: ignore[dict-item]
    med = statistics.median(timings)
    return {
        "median_s": med,
        "min_s": min(timings),
        "max_s": max(timings),
        "stdev_s": statistics.pstdev(timings) if len(timings) > 1 else 0.0,
    }


def write_json(report: RunReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env": report.env,
        "cells": report.cells,
        "runner_setup_s": report.runner_setup_s,
    }
    path.write_text(json.dumps(payload, indent=2))


def cell_to_dict(cell: CellResult) -> dict[str, Any]:
    return asdict(cell)
