#!/usr/bin/env python3
"""Plot the layerwise CAA cos-to-null comparison for two models on one axis.

Reads the per-layer JSONs produced by layer_sweep_cos_to_null.py --save-json
(and the Gemma reference under results/gemma_reference/) and overlays them on a
NORMALISED-DEPTH x-axis so models with different layer counts line up.

Panel A: persona-mean cos-to-null (solid) + nonsense control (dashed) + a faint
         band down to persona-min, per model. Shows the monotonic slide and the
         (in)validity of the control.
Panel B: separation gap = nonsense - persona-mean, per model. The layer the paper
         actually extracts at is where this is healthy but persona-min hasn't yet
         gone strongly negative (sign-flip collapse).

Usage:
    python scripts/plot_layer_comparison.py \
      --json results/qwen_reference/layer_sweep_qwen.json \
      --json results/gemma_reference/layer_sweep_gemma.json \
      --mark Qwen2.5-7B-Instruct:19 --mark gemma-2-27b-it:22
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layerwise cos-to-null comparison across models")
    p.add_argument("--json", action="append", required=True,
                   help="per-layer sweep JSON (repeat for each model)")
    p.add_argument("--mark", action="append", default=[],
                   help="MODEL:LAYER to annotate as the operating point (repeatable)")
    p.add_argument("--output", default=str(REPO_ROOT / "results" / "layer_comparison" / "qwen_vs_gemma_layers.png"))
    return p.parse_args()


def load(path: str) -> dict:
    d = json.load(open(path))
    rows = d["rows"]
    d["_depth"] = np.array([r["depth"] for r in rows], float)
    d["_layer"] = np.array([r["layer"] for r in rows], int)
    d["_pm"] = np.array([r["persona_mean"] for r in rows], float)
    d["_pn"] = np.array([r["persona_min"] for r in rows], float)
    d["_ns"] = np.array([np.nan if r["nonsense"] is None else r["nonsense"] for r in rows], float)
    d["_gap"] = np.array([np.nan if r["gap"] is None else r["gap"] for r in rows], float)
    return d


def main() -> None:
    args = parse_args()
    models = [load(p) for p in args.json]
    marks = {m.split(":")[0]: int(m.split(":")[1]) for m in args.mark}

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})

    for i, d in enumerate(models):
        c = COLORS[i % len(COLORS)]
        name = d.get("model", f"model{i}")
        x = d["_depth"]
        axA.plot(x, d["_pm"], "-", color=c, lw=2.2, label=f"{name} — persona mean")
        axA.plot(x, d["_ns"], "--", color=c, lw=1.4, alpha=0.9, label=f"{name} — nonsense (control)")
        axA.fill_between(x, d["_pn"], d["_pm"], color=c, alpha=0.10, lw=0)
        axB.plot(x, d["_gap"], "-", color=c, lw=2.0, label=name)

        # operating-point marker
        if name in marks:
            L = marks[name]
            j = int(np.where(d["_layer"] == L)[0][0])
            xm, ym = x[j], d["_pm"][j]
            for ax in (axA, axB):
                ax.axvline(xm, color=c, ls=":", lw=1.1, alpha=0.7)
            axA.scatter([xm], [ym], s=90, color=c, edgecolor="white", zorder=6)
            axA.annotate(f"L{L}  (depth {xm:.2f})\ncos={ym:.2f}, min={d['_pn'][j]:.2f}",
                         (xm, ym), textcoords="offset points", xytext=(8, 10),
                         fontsize=8, color=c, fontweight="bold")

    axA.axhline(0.0, color="#888", lw=0.8, ls="-", alpha=0.6)
    axA.axhline(1.0, color="#1f7a1f", lw=0.8, ls=":", alpha=0.5)
    axA.set_ylabel(r"$\cos(\mathbf{v}_{T,c},\,\mathbf{v}_{T,\mathrm{null}})$  (mean over 10×8)")
    axA.set_ylim(-0.3, 1.03)
    axA.set_title("Layerwise CAA cos-to-null: persona-conditioned trait vectors vs null\n"
                  "(both models slide monotonically; Qwen's transition is shifted deeper)")
    axA.legend(loc="lower left", fontsize=8, frameon=False, ncol=len(models))
    axA.grid(alpha=0.2, ls=":")

    axB.axhline(0.0, color="#888", lw=0.8, alpha=0.6)
    axB.set_ylabel("separation gap\n(nonsense − persona)")
    axB.set_xlabel("normalised depth   (layer / (n_layers − 1))")
    axB.legend(loc="upper left", fontsize=8, frameon=False)
    axB.grid(alpha=0.2, ls=":")

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
