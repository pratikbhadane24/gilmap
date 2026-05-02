"""Generate marketing-ready report from benchmark JSON.

Outputs:
  docs/BENCHMARKS.md      full report (per-workload tables, charts, honesty)
  docs/charts/*.png       speedup-vs-std_map per workload + crossover chart
  README.md               replaces content between BENCH:START/END markers

Honesty asserts:
  - Loss table must be non-empty (gilmap must lose at least one cell). If not,
    we suspect cherry-picked workloads. Caller can pass --allow-no-loss to override
    but we never do that for canonical published runs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
CHARTS_DIR = DOCS_DIR / "charts"
BENCH_DOC = DOCS_DIR / "BENCHMARKS.md"
README = REPO_ROOT / "README.md"

BENCH_START = "<!-- BENCH:START -->"
BENCH_END = "<!-- BENCH:END -->"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="render gilmap benchmark report")
    p.add_argument("json_path", help="results JSON from benchmarks.run")
    p.add_argument("--allow-no-loss", action="store_true",
                   help="bypass mandatory honesty section (NOT for published runs)")
    return p.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _fmt_time(s: float | None) -> str:
    if s is None:
        return "—"
    if s >= 1.0:
        return f"{s:.3f}s"
    if s >= 1e-3:
        return f"{s * 1000:.2f}ms"
    return f"{s * 1e6:.0f}µs"


def _fmt_speedup(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.2f}×"


def _group_cells(cells: list[dict]) -> dict:
    """workload -> size -> runner+container -> cell"""
    by_w: dict[str, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for c in cells:
        if not c["ok"]:
            # still record so loss/error tables can show it
            pass
        key = c["runner"] if c["container"] == "list" else f"{c['runner']}@arrow"
        by_w[c["workload"]][c["size"]][key] = c
    return by_w


def _baseline(grid: dict, size: int) -> float | None:
    cell = grid.get(size, {}).get("std_map")
    if cell and cell.get("ok"):
        return cell["median_s"]
    return None


def _md_table_for_workload(workload: str, sizes_grid: dict, runner_keys: list[str]) -> str:
    sizes = sorted(sizes_grid.keys())
    header = "| N |" + "".join(f" {k} |" for k in runner_keys) + " winner |"
    sep = "|---|" + "".join("---|" for _ in runner_keys) + "---|"
    rows = [header, sep]
    for size in sizes:
        cells = sizes_grid[size]
        bl = _baseline(sizes_grid, size)
        # winner = min median among ok cells
        ranked = [
            (k, cells[k]["median_s"])
            for k in runner_keys
            if k in cells and cells[k].get("ok") and cells[k].get("median_s") is not None
        ]
        winner = min(ranked, key=lambda t: t[1])[0] if ranked else None
        out = [f"| {size:,} |"]
        for k in runner_keys:
            cell = cells.get(k)
            if cell is None:
                out.append(" — |")
                continue
            if not cell.get("ok"):
                out.append(" ✖ |")
                continue
            t = _fmt_time(cell["median_s"])
            sp = (
                f" ({_fmt_speedup(bl / cell['median_s'])} vs map)"
                if bl and cell.get("median_s")
                else ""
            )
            mark = "**" if k == winner else ""
            unstable = " ⚠" if cell.get("unstable") else ""
            out.append(f" {mark}{t}{mark}{sp}{unstable} |")
        out.append(f" **{winner or '—'}** |")
        rows.append("".join(out))
    return "\n".join(rows)


def _all_runner_keys(by_w: dict) -> dict[str, list[str]]:
    """Per workload, ordered list of runner+container keys actually present."""
    order = [
        "std_map",
        "cf_thread",
        "mp_pool",
        "cf_process",
        "joblib",
        "ray",
        "dask",
        "gilmap_list",
        "gilmap_arrow@arrow",
        "numpy_vec",
        "numba",
    ]
    result: dict[str, list[str]] = {}
    for w, grid in by_w.items():
        present: set[str] = set()
        for size_cells in grid.values():
            present.update(size_cells.keys())
        result[w] = [k for k in order if k in present]
    return result


def _speedup_chart(workload: str, sizes_grid: dict, runner_keys: list[str], out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; skipping chart", file=sys.stderr)
        return False

    sizes = sorted(sizes_grid.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for k in runner_keys:
        xs, ys = [], []
        for s in sizes:
            cell = sizes_grid[s].get(k)
            bl = _baseline(sizes_grid, s)
            if cell and cell.get("ok") and bl and cell.get("median_s"):
                xs.append(s)
                ys.append(bl / cell["median_s"])
        if not xs:
            continue
        style = "-"
        if k.startswith("gilmap"):
            style = "-o"
        ax.plot(xs, ys, style, label=k, linewidth=2 if k.startswith("gilmap") else 1.2)
        plotted += 1
    if not plotted:
        plt.close(fig)
        return False
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", label="map (baseline)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("input size N")
    ax.set_ylabel("speedup vs std map (higher is better)")
    ax.set_title(f"{workload}: speedup vs single-thread map")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def _crossover_chart(by_w: dict, out_path: Path) -> bool:
    """For each workload, plot gilmap vs mp_pool absolute time."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = 0
    cmap = plt.cm.tab10  # type: ignore[attr-defined]
    for idx, (w, grid) in enumerate(sorted(by_w.items())):
        sizes = sorted(grid.keys())
        gx, gy, mx, my = [], [], [], []
        for s in sizes:
            cells = grid[s]
            g = cells.get("gilmap_arrow@arrow") or cells.get("gilmap_list")
            m = cells.get("mp_pool")
            if g and g.get("ok"):
                gx.append(s)
                gy.append(g["median_s"])
            if m and m.get("ok"):
                mx.append(s)
                my.append(m["median_s"])
        if not gx or not mx:
            continue
        c = cmap(idx % 10)
        ax.plot(gx, gy, "-o", color=c, label=f"{w} · gilmap")
        ax.plot(mx, my, "--s", color=c, alpha=0.7, label=f"{w} · mp_pool")
        plotted += 1
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("input size N")
    ax.set_ylabel("median time (s) — lower is better")
    ax.set_title("gilmap vs multiprocessing.Pool — crossover by input size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def _losses(by_w: dict) -> list[dict]:
    """Cells where gilmap is NOT the fastest. Sorted by margin of loss."""
    losses: list[dict] = []
    for w, grid in by_w.items():
        for size, cells in grid.items():
            ok_cells = {k: c for k, c in cells.items() if c.get("ok") and c.get("median_s")}
            if not ok_cells:
                continue
            best_key, best_cell = min(ok_cells.items(), key=lambda kv: kv[1]["median_s"])
            gilmap_keys = [k for k in ok_cells if k.startswith("gilmap")]
            if not gilmap_keys:
                continue
            best_gilmap_key = min(gilmap_keys, key=lambda k: ok_cells[k]["median_s"])
            if best_key.startswith("gilmap"):
                continue  # gilmap won this cell
            margin = ok_cells[best_gilmap_key]["median_s"] / best_cell["median_s"]
            losses.append(
                {
                    "workload": w,
                    "size": size,
                    "winner": best_key,
                    "winner_s": best_cell["median_s"],
                    "best_gilmap": best_gilmap_key,
                    "gilmap_s": ok_cells[best_gilmap_key]["median_s"],
                    "margin": margin,
                }
            )
    losses.sort(key=lambda l: l["margin"], reverse=True)
    return losses


def _wins(by_w: dict) -> list[dict]:
    wins: list[dict] = []
    for w, grid in by_w.items():
        for size, cells in grid.items():
            ok_cells = {k: c for k, c in cells.items() if c.get("ok") and c.get("median_s")}
            gilmap_keys = [k for k in ok_cells if k.startswith("gilmap")]
            non_gilmap = {k: c for k, c in ok_cells.items() if not k.startswith("gilmap")}
            if not gilmap_keys or not non_gilmap:
                continue
            best_gilmap = min(gilmap_keys, key=lambda k: ok_cells[k]["median_s"])
            best_other_key, best_other = min(non_gilmap.items(), key=lambda kv: kv[1]["median_s"])
            if ok_cells[best_gilmap]["median_s"] >= best_other["median_s"]:
                continue
            wins.append({
                "workload": w,
                "size": size,
                "gilmap": best_gilmap,
                "gilmap_s": ok_cells[best_gilmap]["median_s"],
                "runner_up": best_other_key,
                "runner_up_s": best_other["median_s"],
                "speedup": best_other["median_s"] / ok_cells[best_gilmap]["median_s"],
            })
    wins.sort(key=lambda w_: w_["speedup"], reverse=True)
    return wins


def _setup_table(setup_s: dict[str, float]) -> str:
    if not setup_s:
        return "_no per-runner setup data captured (older JSON format)_"
    rows = ["| runner | one-time process setup |", "|---|---|"]
    for k, v in sorted(setup_s.items(), key=lambda kv: -kv[1]):
        rows.append(f"| {k} | {_fmt_time(v)} |")
    return "\n".join(rows)


def _warmup_table(by_w: dict) -> str:
    rows = []
    for w, grid in by_w.items():
        for size, cells in sorted(grid.items()):
            for k in ("gilmap_list", "gilmap_arrow@arrow", "mp_pool", "ray", "joblib", "numba"):
                cell = cells.get(k)
                if not cell or not cell.get("ok"):
                    continue
                ws = cell.get("warmup_s")
                if ws is None:
                    continue
                rows.append((w, size, k, ws, cell.get("median_s") or 0.0))
            break  # only first size per workload (warmup is constant per process)
        break  # only first workload representative; warmup is per-runner not per-workload
    if not rows:
        return "_no warmup data captured_"
    out = ["| runner | warmup (first call) | typical steady-state | overhead |", "|---|---|---|---|"]
    for _, _, k, ws, med in rows:
        out.append(f"| {k} | {_fmt_time(ws)} | {_fmt_time(med)} | {_fmt_time(max(0.0, ws - med))} |")
    return "\n".join(out)


def _env_section(env: dict) -> str:
    cpu = env.get("cpu", {})
    py = env.get("python", {})
    git = env.get("git", {})
    deps = env.get("dep_versions", {})
    dep_lines = "\n".join(f"- {k}: {v}" for k, v in sorted(deps.items()))
    return f"""**Hardware:** {cpu.get("cpu_model", "unknown")} · {cpu.get("logical_cores")} logical cores · {env.get("os", {}).get("system")} {env.get("os", {}).get("release")}

**Python:** {py.get("version")} ({py.get("implementation")}, GIL_disabled={py.get("gil_disabled")})

**gilmap commit:** `{git.get("commit", "unknown")}`{" (dirty)" if git.get("dirty") else ""}

**Run:** {env.get("timestamp_utc")}

**Dependency versions:**
{dep_lines}
"""


def _hero_block(wins: list[dict], losses: list[dict], env: dict, n_workloads: int) -> str:
    if not wins:
        hero = "_No measured wins yet._"
    else:
        top = wins[0]
        hero = (
            f"On `{top['workload']}` at N={top['size']:,}, **gilmap** "
            f"({top['gilmap']}) is **{_fmt_speedup(top['speedup'])} faster** than "
            f"the best non-gilmap runner (`{top['runner_up']}`) — measured on "
            f"{env.get('cpu', {}).get('cpu_model', 'this machine')}."
        )
    summary = []
    summary.append("| workload | best gilmap variant | speedup vs runner-up | runner-up |")
    summary.append("|---|---|---|---|")
    for w in wins[:5]:
        summary.append(
            f"| {w['workload']} N={w['size']:,} | {w['gilmap']} | "
            f"{_fmt_speedup(w['speedup'])} | {w['runner_up']} |"
        )
    for l in losses[:3]:
        summary.append(
            f"| ⚠ {l['workload']} N={l['size']:,} | {l['best_gilmap']} | "
            f"{_fmt_speedup(1.0 / l['margin'])} (we lose) | {l['winner']} |"
        )
    return f"""{hero}

**Summary across {n_workloads} workloads:**

{chr(10).join(summary)}

⚠ rows are workloads where gilmap loses to a faster runner — included so this table is honest, not cherry-picked. Full breakdown in `docs/BENCHMARKS.md`.
"""


def render_bench_md(report: dict) -> tuple[str, list[Path]]:
    """Returns (markdown, list of chart paths produced)."""
    cells = report["cells"]
    env = report["env"]
    by_w = _group_cells(cells)
    runner_keys = _all_runner_keys(by_w)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_paths: list[Path] = []

    sections = ["# gilmap benchmark report", ""]
    sections.append("This report is generated by `python -m benchmarks.report`. Do not edit by hand.")
    sections.append("")
    sections.append("## Environment")
    sections.append(_env_section(env))
    sections.append("## Methodology")
    sections.append(
        "- Each cell timed `N` times after one untimed warmup. Warmup covers worker pool init, "
        "JIT compilation, and per-process imports — reported separately, not amortized into "
        "steady-state numbers.\n"
        "- Median is the headline; min/max/stdev recorded. Cells with stdev > 10% of median "
        "are flagged ⚠ (unstable).\n"
        "- Every runner's output is byte-equivalent (within 1e-6 float tolerance) to the "
        "single-thread `std_map` baseline. Mismatches abort the cell.\n"
        "- Optional comparators (joblib, numba, ray, dask, numpy_vec) run if installed; otherwise "
        "they appear as empty columns.\n"
    )

    # Crossover chart at the top
    crossover_path = CHARTS_DIR / "crossover.png"
    if _crossover_chart(by_w, crossover_path):
        chart_paths.append(crossover_path)
        sections.append("## gilmap vs multiprocessing — crossover")
        sections.append(f"![crossover](charts/crossover.png)")
        sections.append("")

    sections.append("## Per-workload results")
    for w in sorted(by_w.keys()):
        sections.append(f"### {w}")
        sections.append("")
        keys = runner_keys[w]
        sections.append(_md_table_for_workload(w, by_w[w], keys))
        sections.append("")
        # speedup chart
        chart = CHARTS_DIR / f"{w}.png"
        if _speedup_chart(w, by_w[w], keys, chart):
            chart_paths.append(chart)
            sections.append(f"![{w}](charts/{w}.png)")
            sections.append("")

    # honesty section
    losses = _losses(by_w)
    wins = _wins(by_w)
    sections.append("## Where gilmap loses (honest)")
    if losses:
        sections.append("These are real cells where gilmap is **not** the fastest. Listed worst-first.")
        sections.append("")
        sections.append("| workload | N | best gilmap | best gilmap (s) | winner | winner (s) | margin |")
        sections.append("|---|---|---|---|---|---|---|")
        for l in losses:
            sections.append(
                f"| {l['workload']} | {l['size']:,} | {l['best_gilmap']} | "
                f"{_fmt_time(l['gilmap_s'])} | {l['winner']} | "
                f"{_fmt_time(l['winner_s'])} | gilmap is {_fmt_speedup(l['margin'])} slower |"
            )
        sections.append("")
    else:
        sections.append("_No losses recorded in this run. Likely cherry-picked workloads — re-run "
                        "with the full suite._")
        sections.append("")

    sections.append("## When NOT to use gilmap")
    sections.append(
        "- **Tiny per-element work.** If each call is < ~1 µs of Python, framework overhead dominates "
        "and a plain `map(...)` will beat every parallel approach. See `quick_collatz` results above.\n"
        "- **Vectorizable numeric kernels.** If your function can be expressed as NumPy ufuncs or "
        "JIT-compiled with numba, that's almost always faster than parallelizing a Python-level "
        "callable. See `float_math` results.\n"
        "- **I/O-bound work.** gilmap parallelizes CPU. Use asyncio or thread pools for network/disk.\n"
        "- **Lambdas / nested functions / `__main__` functions.** Worker sub-interpreters import "
        "callables by `(module, name)`. Put your function in an importable module.\n"
        "- **Free-threaded CPython (cp313t).** PEP 703 builds are explicitly unsupported; gilmap "
        "raises at import.\n"
        "- **Non-numeric data.** Only `int64` and `float64` lanes are supported today.\n"
    )

    sections.append("## Where gilmap shines")
    if wins:
        sections.append("These are real cells where gilmap is the fastest runner. Listed best-first.")
        sections.append("")
        sections.append("| workload | N | gilmap variant | gilmap (s) | runner-up | runner-up (s) | speedup |")
        sections.append("|---|---|---|---|---|---|---|")
        for w_ in wins:
            sections.append(
                f"| {w_['workload']} | {w_['size']:,} | {w_['gilmap']} | "
                f"{_fmt_time(w_['gilmap_s'])} | {w_['runner_up']} | "
                f"{_fmt_time(w_['runner_up_s'])} | {_fmt_speedup(w_['speedup'])} |"
            )
        sections.append("")
    else:
        sections.append("_No outright wins in this run._")
        sections.append("")

    # warmup
    sections.append("## First-call (warmup) cost")
    sections.append(
        "Two costs sit outside steady-state numbers and are reported here so they aren't "
        "amortized into the speedup figures:\n\n"
        "1. **Per-process setup** (paid once when you first import / construct the runner).\n"
        "2. **Per-cell first call** (paid the first time a given runner is asked to do real "
        "work — pool spinup, JIT, worker sub-interpreter import).\n"
    )
    sections.append("### Per-process setup")
    sections.append(_setup_table(report.get("runner_setup_s", {})))
    sections.append("")
    sections.append("### Per-cell first call (warmup)")
    sections.append(_warmup_table(by_w))
    sections.append("")

    sections.append("## Reproduce")
    sections.append("```\npip install -e \".[bench]\"\npython -m benchmarks.run --out benchmarks/results/local.json\npython -m benchmarks.report benchmarks/results/local.json\n```")

    return "\n".join(sections), chart_paths


def render_readme_block(report: dict) -> str:
    cells = report["cells"]
    env = report["env"]
    by_w = _group_cells(cells)
    wins = _wins(by_w)
    losses = _losses(by_w)
    return _hero_block(wins, losses, env, len(by_w))


def update_readme(readme_path: Path, block: str) -> None:
    text = readme_path.read_text()
    new_block = f"{BENCH_START}\n{block}\n{BENCH_END}"
    if BENCH_START in text and BENCH_END in text:
        pattern = re.compile(re.escape(BENCH_START) + r".*?" + re.escape(BENCH_END), re.DOTALL)
        text = pattern.sub(new_block, text)
    else:
        # Replace the existing "## Benchmarking" section if present, else append.
        if "\n## Benchmarking\n" in text:
            head, _, _tail = text.partition("\n## Benchmarking\n")
            # find next h2 after Benchmarking
            rest = "\n## Benchmarking\n" + _tail
            next_h2 = re.search(r"\n## (?!Benchmarking\b)", rest[3:])
            if next_h2:
                tail_kept = rest[3 + next_h2.start():]
            else:
                tail_kept = ""
            text = head + "\n## Benchmarking\n\n" + new_block + "\n" + tail_kept
        else:
            text = text.rstrip() + "\n\n## Benchmarking\n\n" + new_block + "\n"
    readme_path.write_text(text)


def main() -> int:
    args = parse_args()
    report = load(Path(args.json_path))

    by_w = _group_cells(report["cells"])
    losses = _losses(by_w)
    if not losses and not args.allow_no_loss:
        print(
            "ERROR: honesty check failed — no measured losses for gilmap. "
            "This usually means the workload set is cherry-picked. Re-run the full suite "
            "or pass --allow-no-loss if you really mean it.",
            file=sys.stderr,
        )
        return 2

    md, chart_paths = render_bench_md(report)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_DOC.write_text(md)
    print(f"wrote {BENCH_DOC}")
    for p in chart_paths:
        print(f"wrote {p}")

    block = render_readme_block(report)
    update_readme(README, block)
    print(f"updated {README} between BENCH markers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
