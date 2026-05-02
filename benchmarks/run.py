"""CLI: python -m benchmarks.run --out path.json [--quick] [--repeats N]"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pyarrow as pa

from . import workloads
from .harness import (
    CellResult,
    RunReport,
    _equal,
    capture_env,
    cell_to_dict,
    summarize,
    time_cell,
    write_json,
)
from .runners import build_runners
from .suite import SCENARIOS, make_inputs

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="gilmap benchmark sweep")
    p.add_argument("--out", required=True, help="JSON output path")
    p.add_argument("--quick", action="store_true", help="single small size, N=1, smoke")
    p.add_argument("--repeats", type=int, default=3, help="timed runs per cell (default 3)")
    p.add_argument("--only", nargs="*", default=None, help="restrict to runner names")
    p.add_argument("--workloads", nargs="*", default=None, help="restrict to workload names")
    p.add_argument(
        "--no-arrow", action="store_true", help="skip Arrow container (gilmap_arrow runner)"
    )
    return p.parse_args(argv)


def _resolve_func(name: str):
    return getattr(workloads, name)


def _make_data(workload: str, size: int, lane: str, container: str) -> Any:
    raw = make_inputs(workload, size, lane)
    if container == "list":
        return raw
    if container == "arrow":
        return pa.array(raw)
    raise ValueError(container)


def _eq_check(baseline_out: Any, runner_out: Any) -> bool:
    return _equal(baseline_out, runner_out)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repeats = 1 if args.quick else args.repeats
    runners = build_runners(args.only)
    report = RunReport(env=capture_env(REPO_ROOT))

    scenarios = list(SCENARIOS)
    if args.workloads:
        scenarios = [s for s in scenarios if s.workload in args.workloads]

    # Filter runners by availability once.
    active: list = []
    for r in runners:
        ok, reason = r.available()
        if not ok:
            print(f"[skip runner] {r.name}: {reason}", file=sys.stderr)
            continue
        active.append(r)

    # Setup all runners once — pool spinup, JIT compile, ray.init, etc.
    # Steady-state cells then exclude these costs. First-call warmup_s
    # captures the *first timed* call's overhead per cell.
    runner_warmup_global: dict[str, float] = {}
    setup_failed: dict[str, str] = {}
    for r in active:
        t0 = time.perf_counter()
        try:
            r.setup()
            runner_warmup_global[r.name] = time.perf_counter() - t0
            print(f"[setup] {r.name}: {runner_warmup_global[r.name]:.3f}s")
        except Exception as exc:
            setup_failed[r.name] = str(exc)
            print(f"[setup-fail] {r.name}: {exc}", file=sys.stderr)
    report.runner_setup_s = runner_warmup_global

    try:
        for sc in scenarios:
            sizes = (sc.quick_size,) if args.quick else sc.sizes
            func = _resolve_func(sc.workload)

            for size in sizes:
                list_data = _make_data(sc.workload, size, sc.lane, "list")
                arrow_data = None
                if not args.no_arrow:
                    arrow_data = _make_data(sc.workload, size, sc.lane, "arrow")

                # Baseline (std_map) first — provides expected output for equality.
                baseline_out: Any = None
                baseline = next((r for r in active if r.name == "std_map"), None)
                if baseline and baseline.name not in setup_failed:
                    timings, _w, baseline_out = time_cell(
                        lambda: baseline.run(func, list_data), n=repeats, warmup=False
                    )
                    stats = summarize(timings)
                    report.cells.append(cell_to_dict(CellResult(
                        workload=sc.workload, size=size, lane=sc.lane,
                        runner=baseline.name, container="list", n=repeats,
                        median_s=stats["median_s"], min_s=stats["min_s"],
                        max_s=stats["max_s"], stdev_s=stats["stdev_s"],
                        warmup_s=None, ok=True, unstable=_unstable(stats),
                    )))
                    print(f"[ok] {sc.workload} N={size} std_map  median={stats['median_s']:.4f}s")

                for r in active:
                    if r.name == "std_map":
                        continue
                    for container, data in (("list", list_data), ("arrow", arrow_data)):
                        if data is None:
                            continue
                        if not r.supports(sc.workload, container):
                            continue
                        if r.name in setup_failed:
                            report.cells.append(cell_to_dict(CellResult(
                                workload=sc.workload, size=size, lane=sc.lane,
                                runner=r.name, container=container, n=0,
                                median_s=None, min_s=None, max_s=None,
                                stdev_s=None, warmup_s=None, ok=False,
                                skipped=True, skip_reason=setup_failed[r.name],
                                error=f"setup-fail: {setup_failed[r.name]}",
                            )))
                            continue
                        cell = _run_one_no_setup(
                            r, func, data, sc, size, container,
                            baseline_out, repeats,
                            global_warmup=runner_warmup_global.get(r.name),
                        )
                        report.cells.append(cell_to_dict(cell))
    finally:
        for r in active:
            try:
                r.teardown()
            except Exception:
                pass

    out_path = Path(args.out)
    write_json(report, out_path)
    print(f"\nwrote {out_path} ({len(report.cells)} cells)")
    return 0


def _unstable(stats: dict[str, Any]) -> bool:
    med = stats.get("median_s")
    sd = stats.get("stdev_s")
    if not med or sd is None:
        return False
    return sd > 0.1 * med


def _run_one_no_setup(
    runner, func, data, sc, size, container, baseline_out, repeats,
    *, global_warmup: float | None,
) -> CellResult:
    """Run cell. Setup already done globally. Per-cell warmup is one timed
    no-record run that exposes per-cell first-call cost (e.g. mp_pool may
    fork on first call even though Pool() returned earlier)."""
    name = runner.name
    try:
        timings, warmup_s, last_out = time_cell(
            lambda: runner.run(func, data), n=repeats, warmup=True
        )
    except Exception as exc:
        traceback.print_exc()
        return CellResult(
            workload=sc.workload, size=size, lane=sc.lane, runner=name,
            container=container, n=0, median_s=None, min_s=None, max_s=None,
            stdev_s=None, warmup_s=None, ok=False, error=str(exc),
        )

    ok = _eq_check(baseline_out, last_out) if baseline_out is not None else True
    stats = summarize(timings)
    unstable = _unstable(stats)
    flag = "ok" if ok else "MISMATCH"
    eff_warmup = global_warmup if (global_warmup and global_warmup > (warmup_s or 0)) else warmup_s
    print(
        f"[{flag}] {sc.workload} N={size} {name}/{container}  median={stats['median_s']:.4f}s warmup={warmup_s:.3f}s"
    )
    return CellResult(
        workload=sc.workload, size=size, lane=sc.lane, runner=name,
        container=container, n=repeats,
        median_s=stats["median_s"], min_s=stats["min_s"],
        max_s=stats["max_s"], stdev_s=stats["stdev_s"],
        warmup_s=eff_warmup, ok=ok, unstable=unstable,
        error="" if ok else "result mismatch vs std_map baseline",
    )


if __name__ == "__main__":
    raise SystemExit(main())
