#!/usr/bin/env python3
# analyze_results.py

"""
  Figure 1  – Scalability: Mean time vs N
  Figure 2  – Scalability: Effective GB/s vs N
  Figure 3  – Scalability: Speedup vs N
  Figure 4  – Roofline chart (T4 + AI markers)
  Figure 5  – Block-size sweep: time vs config
  Figure 6  – Block-size sweep: speedup vs blocks
  Figure 7  – Weak scaling: time vs num_blocks
  Figure 8  – Weak scaling: SS vs num_blocks
  Figure 9  – Comm overhead: bar chart
"""

import os
import re
import sys
import math

import argparse
import collections

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ─────────────────────────────────────────────────────────────────────────────
# NVIDIA T4 hardware constants (for roofline overlay)
# ─────────────────────────────────────────────────────────────────────────────
T4_PEAK_FP64_TFLOPS  = 0.260          # TFLOPS  (FP64 matrix)
T4_PEAK_BW_GBS       = 320.0          # GB/s    (HBM)
T4_RIDGE_FLOP_BYTE   = (T4_PEAK_FP64_TFLOPS * 1e12) / (T4_PEAK_BW_GBS * 1e9)

# Jacobi stencil arithmetic intensity
# FLOPs/point = 6  (4 adds + 1 mul h²f + 1 mul ×0.25)
# Streaming BW = 3 arrays × 8 B = 24 B/point  (Williams et al. 2009)
AI_V1 = 6.0 / 24.0    # 0.25 FLOP/byte  (global mem only)
AI_V2 = 6.0 / 26.1    # ≈0.23 FLOP/byte (shared mem tile: (18²/256 + 2)×8 B)
AI_V3 = AI_V2          # same memory pattern as V2

COLORS = {
    "CPU": "#555555",
    "V1":  "#2196F3",   # blue
    "V2":  "#4CAF50",   # green
    "V3":  "#FF5722",   # orange-red
}

MARKERS = {"CPU": "s", "V1": "o", "V2": "^", "V3": "D"}
LINESTYLES = {"CPU": "--", "V1": "-", "V2": "-", "V3": "-"}

OUTDIR = "plots"

# ─────────────────────────────────────────────────────────────────────────────
# CSV parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_csv(lines):
    """
    Returns a dict:
      data["scalability"]    = list of dicts
      data["block_sweep"]    = list of dicts
      data["block_sweep_v2"] = list of dicts
      data["block_sweep_v3"] = list of dicts
      data["weak_scaling"]   = list of dicts
      data["comm_overhead"]  = list of dicts
    """
    data = collections.defaultdict(list)

    for raw in lines:
        line = raw.strip()
        if not line.startswith("BENCH_CSV"):
            continue
        # strip leading "BENCH_CSV,"
        parts = line.split(",")
        if len(parts) < 2:
            continue
        tag = parts[1]

        try:
            if tag == "scalability" and len(parts) >= 13:
                # BENCH_CSV,scalability,N,solver,bx,by,num_blocks,
                #           mean_ms,std_ms,gb_s,gflops,speedup_cpu,speedup_v1
                data["scalability"].append({
                    "N":           int(parts[2]),
                    "solver":      parts[3],
                    "bx":          int(parts[4]),
                    "by":          int(parts[5]),
                    "num_blocks":  int(parts[6]),
                    "mean_ms":     float(parts[7]),
                    "std_ms":      float(parts[8]),
                    "gb_s":        float(parts[9]),
                    "gflops":      float(parts[10]),
                    "speedup_cpu": float(parts[11]),
                    "speedup_v1":  float(parts[12]),
                })

            elif tag in ("block_sweep", "block_sweep_v2", "block_sweep_v3") \
                    and len(parts) >= 12:
                # BENCH_CSV, block_sweep, N, bx, by, num_blocks,
                #           mean_ms, std_ms, gb_s, gflops, speedup_ref16x16, speedup_cpu
                data[tag].append({
                    "N":                int(parts[2]),
                    "bx":               int(parts[3]),
                    "by":               int(parts[4]),
                    "num_blocks":       int(parts[5]),
                    "mean_ms":          float(parts[6]),
                    "std_ms":           float(parts[7]),
                    "gb_s":             float(parts[8]),
                    "gflops":           float(parts[9]),
                    "speedup_ref16x16": float(parts[10]),
                    "speedup_cpu":      float(parts[11]),
                    "label":            f"{parts[3]}×{parts[4]}",
                })

            elif tag == "weak_scaling" and len(parts) >= 9:
                # BENCH_CSV,weak_scaling,num_blocks,N,solver,
                #           mean_ms,std_ms,gb_s,scaled_speedup
                data["weak_scaling"].append({
                    "num_blocks":      int(parts[2]),
                    "N":               int(parts[3]),
                    "solver":          parts[4],
                    "mean_ms":         float(parts[5]),
                    "std_ms":          float(parts[6]),
                    "gb_s":            float(parts[7]),
                    "scaled_speedup":  float(parts[8]),
                })

            elif tag == "comm_overhead" and len(parts) >= 9:
                # BENCH_CSV,comm_overhead,N,strategy,component,
                #           mean_ms,std_ms,overhead_pct,0
                data["comm_overhead"].append({
                    "N":            int(parts[2]),
                    "strategy":     parts[3],
                    "component":    parts[4],
                    "mean_ms":      float(parts[5]),
                    "std_ms":       float(parts[6]),
                    "overhead_pct": float(parts[7]),
                })

        except (IndexError, ValueError) as exc:
            print(f"[WARNING] Could not parse line: {line!r}  ({exc})", file=sys.stderr)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def _solver_key(row):
    """Return 'CPU', 'V1', 'V2', or 'V3' for a row."""
    s = row.get("solver", "")
    if "CPU" in s.upper():
        return "CPU"
    for v in ("V3", "V2", "V1"):
        if v in s:
            return v
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1-3: Scalability
# ─────────────────────────────────────────────────────────────────────────────

def plot_scalability(rows):
    if not rows:
        print("[SKIP] No scalability data.")
        return

    # Group by solver
    by_solver = collections.defaultdict(lambda: {"N": [], "mean_ms": [],
                                                   "std_ms": [], "gb_s": [],
                                                   "gflops": [], "speedup_cpu": []})
    for r in rows:
        sk = _solver_key(r)
        d  = by_solver[sk]
        d["N"].append(r["N"])
        d["mean_ms"].append(r["mean_ms"])
        d["std_ms"].append(r["std_ms"])
        d["gb_s"].append(r["gb_s"])
        d["gflops"].append(r["gflops"])
        d["speedup_cpu"].append(r["speedup_cpu"])

    solvers_ordered = [s for s in ("CPU", "V1", "V2", "V3") if s in by_solver]

    # ── Figure 1: Time vs N ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for sk in solvers_ordered:
        d = by_solver[sk]
        xs = d["N"];  ys = d["mean_ms"];  errs = d["std_ms"]
        ax.errorbar(xs, ys, yerr=errs,
                    color=COLORS.get(sk, "gray"),
                    marker=MARKERS.get(sk, "x"),
                    linestyle=LINESTYLES.get(sk, "-"),
                    label=sk, capsize=4, linewidth=1.8)
    ax.set_xlabel("Grid size N", fontsize=12)
    ax.set_ylabel("Mean time per run [ms]", fontsize=12)
    ax.set_title("Scalability – Time vs N  (BENCH_ITERS=200, runs=20)", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)
    _savefig(fig, "fig1_scalability_time.png")

    # ── Figure 2: GB/s vs N ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for sk in solvers_ordered:
        d = by_solver[sk]
        ax.plot(d["N"], d["gb_s"],
                color=COLORS.get(sk, "gray"),
                marker=MARKERS.get(sk, "x"),
                linestyle=LINESTYLES.get(sk, "-"),
                label=sk, linewidth=1.8)
    ax.axhline(T4_PEAK_BW_GBS, color="red", linestyle=":", linewidth=1.5,
               label=f"T4 peak BW ({T4_PEAK_BW_GBS:.0f} GB/s)")
    ax.set_xlabel("Grid size N", fontsize=12)
    ax.set_ylabel("Effective bandwidth [GB/s]", fontsize=12)
    ax.set_title("Scalability – Bandwidth vs N", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)
    _savefig(fig, "fig2_scalability_bw.png")

    # ── Figure 3: Speedup vs N ────────────────────────────────────────────────
    gpu_solvers = [s for s in solvers_ordered if s != "CPU"]
    if gpu_solvers and by_solver["CPU"]["mean_ms"]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for sk in gpu_solvers:
            d = by_solver[sk]
            ax.plot(d["N"], d["speedup_cpu"],
                    color=COLORS.get(sk, "gray"),
                    marker=MARKERS.get(sk, "x"),
                    linestyle=LINESTYLES.get(sk, "-"),
                    label=f"{sk} vs CPU", linewidth=1.8)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="CPU baseline")
        ax.set_xlabel("Grid size N", fontsize=12)
        ax.set_ylabel("Speedup vs CPU", fontsize=12)
        ax.set_title("Scalability – GPU Speedup vs N", fontsize=12)
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=10)
        _savefig(fig, "fig3_scalability_speedup.png")
    else:
        print("[INFO] Speedup plot skipped (no CPU data).")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Roofline
# ─────────────────────────────────────────────────────────────────────────────

def plot_roofline(scal_rows):
    """
    Draws the T4 roofline and overlays the achieved GB/s for each solver
    at the largest available N.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Roofline lines
    ai_range   = [10**x for x in [i * 0.1 for i in range(-20, 30)]]
    mem_roof   = [T4_PEAK_BW_GBS * ai for ai in ai_range]          # BW-bound
    comp_roof  = [T4_PEAK_FP64_TFLOPS * 1e3 for _ in ai_range]     # compute-bound [GFLOP/s]
    roofline_y = [min(m, c) for m, c in zip(mem_roof, comp_roof)]

    ax.loglog(ai_range, roofline_y, "k-", linewidth=2, label="T4 Roofline")
    ax.axvline(T4_RIDGE_FLOP_BYTE, color="black", linestyle=":", alpha=0.5,
               label=f"Ridge ≈ {T4_RIDGE_FLOP_BYTE:.2f} FLOP/B")

    # Overlay achieved performance (use largest N rows)
    if scal_rows:
        max_N = max(r["N"] for r in scal_rows)
        for r in scal_rows:
            if r["N"] != max_N:
                continue
            sk = _solver_key(r)
            ai = {"V1": AI_V1, "V2": AI_V2, "V3": AI_V3}.get(sk, AI_V1)
            gf = r["gflops"]
            ax.plot(ai, gf,
                    marker=MARKERS.get(sk, "o"), markersize=10,
                    color=COLORS.get(sk, "gray"),
                    label=f"{sk} @ N={max_N}  (AI={ai:.2f})")

    # Arithmetic intensity markers
    for name, ai_val in [("V1 AI", AI_V1), ("V2/V3 AI", AI_V2)]:
        ax.axvline(ai_val, color="navy", linestyle="--", alpha=0.4)
        ax.text(ai_val * 1.05, 0.001, name, fontsize=8, color="navy", rotation=90,
                va="bottom")

    ax.set_xlabel("Arithmetic Intensity [FLOP/byte]", fontsize=12)
    ax.set_ylabel("Performance [GFLOP/s]", fontsize=12)
    ax.set_title(
        "Roofline Model – NVIDIA T4  (FP64)\n"
        f"Peak BW = {T4_PEAK_BW_GBS:.0f} GB/s | "
        f"Peak FP64 = {T4_PEAK_FP64_TFLOPS*1e3:.0f} GFLOP/s | "
        f"Ridge = {T4_RIDGE_FLOP_BYTE:.2f} FLOP/B",
        fontsize=11)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    _savefig(fig, "fig4_roofline.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figures 5-6: Block-size sweep
# ─────────────────────────────────────────────────────────────────────────────

def plot_block_sweep(bs_rows, bs_v2_rows, bs_v3_rows):
    if not bs_rows:
        print("[SKIP] No block_sweep data.")
        return

    # Sort by num_blocks ascending
    bs_rows    = sorted(bs_rows,    key=lambda r: r["num_blocks"])
    bs_v2_rows = sorted(bs_v2_rows, key=lambda r: r["num_blocks"])
    bs_v3_rows = sorted(bs_v3_rows, key=lambda r: r["num_blocks"])

    configs   = [r["label"]      for r in bs_rows]
    n_blocks  = [r["num_blocks"] for r in bs_rows]
    times_v1  = [r["mean_ms"]    for r in bs_rows]
    errs_v1   = [r["std_ms"]     for r in bs_rows]
    sp_v1     = [r["speedup_ref16x16"] for r in bs_rows]

    # ── Figure 5: Time vs block config ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(range(len(configs)))
    ax.bar(x, times_v1, color=COLORS["V1"], alpha=0.85, label="V1 flex")
    ax.errorbar(x, times_v1, yerr=errs_v1, fmt="none",
                color="black", capsize=4, linewidth=1.2)

    # V2 and V3 fixed at 16x16 reference line
    if bs_v2_rows:
        t_v2 = bs_v2_rows[0]["mean_ms"]
        ax.axhline(t_v2, color=COLORS["V2"], linestyle="--",
                   linewidth=1.8, label=f"V2 @ 16×16 ({t_v2:.2f} ms)")
    if bs_v3_rows:
        t_v3 = bs_v3_rows[0]["mean_ms"]
        ax.axhline(t_v3, color=COLORS["V3"], linestyle="-.",
                   linewidth=1.8, label=f"V3 @ 16×16 ({t_v3:.2f} ms)")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Block configuration (bx × by)", fontsize=12)
    ax.set_ylabel("Mean time [ms]", fontsize=12)
    ax.set_title(
        "Block-size sweep (Strong Scaling)\n"
        f"Fixed N={bs_rows[0]['N']}  |  V1 flex kernel  |  runs=20",
        fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.35)
    _savefig(fig, "fig5_block_sweep_time.png")

    # ── Figure 6: Speedup vs num_blocks ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(n_blocks, sp_v1,
            color=COLORS["V1"], marker=MARKERS["V1"],
            linestyle="-", linewidth=1.8, label="V1 flex (vs 16×16 ref)")
    ax.axhline(1.0, color="gray", linestyle="--", label="16×16 reference")
    # Ideal linear speedup curve (optional, for context)
    ref_nb = next((r["num_blocks"] for r in bs_rows
                   if r["bx"] == 16 and r["by"] == 16), None)
    if ref_nb:
        ideal = [ref_nb / nb for nb in n_blocks]
        ax.plot(n_blocks, ideal, color="silver", linestyle=":",
                label="Ideal (1/num_blocks)")
    ax.set_xlabel("Number of thread blocks", fontsize=12)
    ax.set_ylabel("Speedup vs 16×16 reference", fontsize=12)
    ax.set_title("Block-size sweep – Speedup vs num_blocks", fontsize=12)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=10)
    _savefig(fig, "fig6_block_sweep_speedup.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figures 7-8: Weak scaling
# ─────────────────────────────────────────────────────────────────────────────

def plot_weak_scaling(rows):
    if not rows:
        print("[SKIP] No weak_scaling data.")
        return

    by_solver = collections.defaultdict(lambda: {"num_blocks": [], "mean_ms": [],
                                                   "std_ms": [], "ss": []})
    for r in rows:
        sk = _solver_key(r)
        by_solver[sk]["num_blocks"].append(r["num_blocks"])
        by_solver[sk]["mean_ms"].append(r["mean_ms"])
        by_solver[sk]["std_ms"].append(r["std_ms"])
        by_solver[sk]["ss"].append(r["scaled_speedup"])

    solvers = [s for s in ("CPU", "V1", "V2", "V3") if s in by_solver]

    # ── Figure 7: Time vs num_blocks ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for sk in solvers:
        d = by_solver[sk]
        ax.errorbar(d["num_blocks"], d["mean_ms"], yerr=d["std_ms"],
                    color=COLORS.get(sk, "gray"),
                    marker=MARKERS.get(sk, "x"),
                    linestyle=LINESTYLES.get(sk, "-"),
                    label=sk, capsize=4, linewidth=1.8)

    ax.set_xlabel("Number of thread blocks  b  (N = 16√b)", fontsize=12)
    ax.set_ylabel("Mean time per run [ms]", fontsize=12)
    ax.set_title(
        "Weak Scaling – Time vs num_blocks\n"
        "(fix n=256 threads/block; ideal: horizontal line)",
        fontsize=11)
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)
    _savefig(fig, "fig7_weak_scaling_time.png")

    # ── Figure 8: Scaled speedup SS vs num_blocks ─────────────────────────────
    gpu_solvers = [s for s in solvers if s != "CPU"]
    if gpu_solvers:
        fig, ax = plt.subplots(figsize=(7, 5))
        for sk in gpu_solvers:
            d = by_solver[sk]
            # only rows where SS was computed (ss > 0)
            xs = [b for b, s in zip(d["num_blocks"], d["ss"]) if s > 0]
            ys = [s for s in d["ss"] if s > 0]
            if xs:
                ax.plot(xs, ys,
                        color=COLORS.get(sk, "gray"),
                        marker=MARKERS.get(sk, "x"),
                        linestyle="-", label=sk, linewidth=1.8)
        # Ideal: constant (equal to value at smallest b)
        all_ss = [s for sk in gpu_solvers for s in by_solver[sk]["ss"] if s > 0]
        if all_ss:
            ideal_ss = all_ss[0]
            all_nb   = sorted({b for sk in gpu_solvers
                                for b in by_solver[sk]["num_blocks"]})
            ax.axhline(ideal_ss, color="gray", linestyle="--",
                       label=f"Ideal SS = {ideal_ss:.1f}")

        ax.set_xlabel("Number of thread blocks  b", fontsize=12)
        ax.set_ylabel("Scaled speedup  SS(b, n)", fontsize=12)
        ax.set_title(
            "Weak Scaling – Scaled Speedup SS(b,n)\n"
            "SS = T_seq(N) / T_gpu(N)  |  ideal: constant",
            fontsize=11)
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=10)
        _savefig(fig, "fig8_weak_scaling_speedup.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: Communication overhead
# ─────────────────────────────────────────────────────────────────────────────

def plot_comm_overhead(rows):
    if not rows:
        print("[SKIP] No comm_overhead data.")
        return

    # Build a simple summary dict
    summary = {}
    for r in rows:
        key = (r["strategy"], r["component"])
        summary[key] = r

    strat_a_kern  = summary.get(("StratA", "kernel_per_iter"),  {}).get("mean_ms", 0)
    strat_a_trans = summary.get(("StratA", "transfer_per_check"), {}).get("mean_ms", 0)
    strat_b_kern  = summary.get(("StratB", "kernel_per_iter"),  {}).get("mean_ms", 0)
    strat_b_trans = summary.get(("StratB", "transfer_per_check"), {}).get("mean_ms", 0)

    pct_a = summary.get(("StratA", "transfer_per_check"), {}).get("overhead_pct", 0)
    pct_b = summary.get(("StratB", "transfer_per_check"), {}).get("overhead_pct", 0)

    N_val = rows[0]["N"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

    # Left: absolute times
    categories = ["Kernel\n(per iter)", "Transfer\n(per check)"]
    ax = axes[0]
    x = [0, 1]
    width = 0.3
    bars_a = [strat_a_kern, strat_a_trans]
    bars_b = [strat_b_kern, strat_b_trans]
    ax.bar([xi - width/2 for xi in x], bars_a, width, label="Strategy A (V1)", color=COLORS["V1"])
    ax.bar([xi + width/2 for xi in x], bars_b, width, label="Strategy B (V2)", color=COLORS["V2"])
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Time [ms]", fontsize=12)
    ax.set_title(f"Communication Overhead  (N={N_val})", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_yscale("log")

    # Right: overhead % per iteration
    ax2 = axes[1]
    labels = [f"Strat A\n(V1)\n{pct_a:.1f}%", f"Strat B\n(V2)\n{pct_b:.1f}%"]
    heights = [pct_a, pct_b]
    colors  = [COLORS["V1"], COLORS["V2"]]
    ax2.bar(labels, heights, color=colors, alpha=0.85, width=0.4)
    ax2.set_ylabel("Overhead per iteration [%]", fontsize=12)
    ax2.set_title("Transfer overhead vs kernel time\n(amortised over check_every=100 iters)",
                  fontsize=11)
    ax2.grid(True, axis="y", alpha=0.35)

    plt.tight_layout()
    _savefig(fig, "fig9_comm_overhead.png")


# ─────────────────────────────────────────────────────────────────────────────
# Print a text summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(data):
    print("\n" + "=" * 65)
    print("  PARSED DATA SUMMARY")
    print("=" * 65)
    for tag, rows in data.items():
        print(f"  {tag:<20s}  {len(rows):>4d} rows")
    print("=" * 65 + "\n")

    scal = data.get("scalability", [])
    if scal:
        print("  Scalability  (mean_ms,  GB/s,  speedup_cpu)")
        print(f"  {'N':>6}  {'Solver':<8}  {'mean_ms':>10}  {'GB/s':>8}  {'Speedup':>8}")
        print("  " + "-" * 45)
        for r in sorted(scal, key=lambda x: (x["N"], x["solver"])):
            print(f"  {r['N']:>6}  {r['solver']:<8}  {r['mean_ms']:>10.3f}  "
                  f"{r['gb_s']:>8.2f}  {r['speedup_cpu']:>8.2f}x")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parse BENCH_CSV output from minisolver and generate plots.")
    parser.add_argument("input", nargs="?", default="-",
                        help="Input file with BENCH_CSV lines (default: stdin)")
    parser.add_argument("--outdir", default="plots",
                        help="Output directory for PNG files (default: plots/)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Only print summary table, do not save plots")
    args = parser.parse_args()

    global OUTDIR
    OUTDIR = args.outdir

    # Read input
    if args.input == "-":
        lines = sys.stdin.readlines()
    else:
        with open(args.input) as f:
            lines = f.readlines()

    data = parse_csv(lines)
    print_summary(data)

    if args.no_plots:
        return

    if not HAS_MPL:
        print("[ERROR] matplotlib is not installed.  Run:  pip install matplotlib")
        sys.exit(1)

    print(f"Generating plots in ./{OUTDIR}/  ...")

    plot_scalability(data.get("scalability", []))
    plot_roofline(data.get("scalability", []))
    plot_block_sweep(
        data.get("block_sweep", []),
        data.get("block_sweep_v2", []),
        data.get("block_sweep_v3", []),
    )
    plot_weak_scaling(data.get("weak_scaling", []))
    plot_comm_overhead(data.get("comm_overhead", []))

    print("\nDone. All figures saved to:", OUTDIR)


if __name__ == "__main__":
    main()