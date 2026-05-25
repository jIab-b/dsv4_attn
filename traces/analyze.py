"""Run after `modal volume get dsv4-traces /traces ./` to summarize the nsys rep.

This produces:
  - kernel_top.csv        top-N CUDA kernels by total time
  - api_top.csv           top CUDA runtime API calls by total time (host overhead)
  - nvtx_top.csv          top NVTX ranges (per-layer / per-region time)
  - SUMMARY.md            human-readable summary

Usage:
    python analyze.py dsv4_decode.nsys-rep
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def nsys_stats(rep: Path, report: str, out_stem: Path) -> Path:
    out_csv = out_stem.with_suffix(f".{report}.csv")
    subprocess.run(
        [
            "nsys",
            "stats",
            "--report",
            report,
            "--format",
            "csv",
            "--output",
            str(out_stem),
            str(rep),
        ],
        check=True,
    )
    # nsys writes "<stem>_<report>.csv" with the report id appended
    candidates = list(out_stem.parent.glob(f"{out_stem.name}_{report}*.csv"))
    if not candidates:
        candidates = list(out_stem.parent.glob(f"{out_stem.name}*{report}*.csv"))
    assert candidates, f"no csv produced for {report}"
    return candidates[0]


def topN(csv_path: Path, n: int, time_col_candidates: tuple[str, ...]) -> list[dict]:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    cols = rows[0].keys()
    time_col = next((c for c in time_col_candidates if c in cols), None)
    if time_col is None:
        return rows[:n]
    def k(r):
        v = r[time_col].replace(",", "").strip()
        try:
            return float(v)
        except Exception:
            return 0.0
    return sorted(rows, key=k, reverse=True)[:n]


def main(rep_arg: str) -> None:
    rep = Path(rep_arg).resolve()
    assert rep.exists(), f"missing {rep}"
    out_dir = rep.parent
    stem = out_dir / "dsv4_decode_stats"

    if shutil.which("nsys") is None:
        print("nsys not on PATH; install Nsight Systems first", file=sys.stderr)
        sys.exit(2)

    print("running nsys stats …")
    kern_csv = nsys_stats(rep, "cuda_gpu_kern_sum", stem)
    api_csv = nsys_stats(rep, "cuda_api_sum", stem)
    try:
        nvtx_csv = nsys_stats(rep, "nvtx_sum", stem)
    except subprocess.CalledProcessError:
        nvtx_csv = None

    kern_top = topN(kern_csv, 25, ("Total Time (ns)", "Total Time", "Time (ns)"))
    api_top = topN(api_csv, 15, ("Total Time (ns)", "Total Time", "Time (ns)"))
    nvtx_top = topN(nvtx_csv, 20, ("Total Time (ns)", "Total Time")) if nvtx_csv else []

    def fmt_rows(rows: list[dict]) -> str:
        if not rows:
            return "(empty)\n"
        cols = list(rows[0].keys())
        keep = [c for c in cols if c in {"Time (%)", "Time", "Total Time (ns)", "Total Time", "Instances", "Avg (ns)", "Avg", "Name", "Range", "Operation", "Category"}]
        if not keep:
            keep = cols[:6]
        lines = [" | ".join(keep)]
        for r in rows:
            lines.append(" | ".join(str(r.get(c, ""))[:120] for c in keep))
        return "\n".join(lines) + "\n"

    summary = []
    summary.append(f"# DSv4-Flash decode nsys trace summary\n")
    summary.append(f"Source: `{rep.name}` ({rep.stat().st_size/1e6:.1f} MB)\n")
    summary.append("## Top CUDA kernels by total time\n```\n" + fmt_rows(kern_top) + "```\n")
    summary.append("## Top CUDA runtime API calls (host overhead)\n```\n" + fmt_rows(api_top) + "```\n")
    if nvtx_top:
        summary.append("## Top NVTX ranges\n```\n" + fmt_rows(nvtx_top) + "```\n")
    else:
        summary.append("## NVTX ranges\n(none captured)\n")

    (out_dir / "KERNEL_BREAKDOWN.md").write_text("\n".join(summary))
    print(f"wrote {out_dir/'KERNEL_BREAKDOWN.md'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dsv4_decode.nsys-rep")
