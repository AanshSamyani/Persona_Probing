#!/usr/bin/env python3
"""tiv4: figures for the IV trajectory, with skeptical traits flagged.

Skeptical traits (unreliable negative pole on OLMo):
  - honesty: "be deceptive" is refused even by SFT -> stays honest -> contrast ~ noise.
  - warmth / empathy: "be cold/callous" fails for personas whose identity IS the trait.
These are drawn dashed + marked (!) and called out in the caption + SUMMARY.md.

Reads outputs/OLMo-2-1124-7B/iv_trajectory/*, writes results/iv_trajectory/.

Usage:
    python pipeline/tiv4_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from persona_steering.config import OUTPUTS_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = OUTPUTS_DIR / "OLMo-2-1124-7B" / "iv_trajectory"
RESULTS_DIR = REPO_ROOT / "results" / "iv_trajectory"

STAGE_ORDER = ["pretrain_1pct", "pretrain_10pct", "pretrain_50pct", "base", "sft", "dpo", "instruct"]
STAGE_LABEL = {"pretrain_1pct": "1%", "pretrain_10pct": "10%", "pretrain_50pct": "50%",
               "base": "Base", "sft": "SFT", "dpo": "DPO", "instruct": "Instruct"}
SKEPTICAL = {"honesty", "warmth", "empathy"}   # unreliable negative pole on OLMo
TRAIT_LABEL = {"assertiveness": "Assertiveness", "confidence": "Confidence", "deference": "Deference",
               "empathy": "Empathy", "honesty": "Honesty", "impulsivity": "Impulsivity",
               "risk_taking": "Risk-taking", "warmth": "Warmth"}


def load(name):
    return json.loads((TRAJ_DIR / name).read_text())


def sft_boundary(stages):
    return (stages.index("sft") - 0.5) if "sft" in stages else None


def fig_variance_trajectory(var, meta, stages):
    traits = meta["traits"]
    x = range(len(stages))
    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(7, 4.2))
    reliable, skeptic = [], []
    for i, t in enumerate(sorted(traits)):
        ys = [var.get(s, {}).get(t, np.nan) for s in stages]
        if t in SKEPTICAL:
            ax.plot(x, ys, "o--", color=cmap(i % 10), lw=1.4, ms=4, alpha=0.7,
                    label=f"{TRAIT_LABEL.get(t, t)} (!)")
            skeptic.append(ys)
        else:
            ax.plot(x, ys, "o-", color=cmap(i % 10), lw=1.8, ms=5, label=TRAIT_LABEL.get(t, t))
            reliable.append(ys)
    if reliable:
        ax.plot(x, np.nanmean(reliable, axis=0), "k-", lw=2.6, alpha=0.55, label="Mean (reliable)")
    b = sft_boundary(stages)
    if b is not None:
        ax.axvline(b, color="#999", ls="--", lw=1, alpha=0.7)
        ax.text(b + 0.06, ax.get_ylim()[1], "SFT", fontsize=8, color="#666", va="top")
    ax.set_xticks(list(x)); ax.set_xticklabels([STAGE_LABEL.get(s, s) for s in stages], fontsize=9)
    ax.set_xlabel("Training Stage", fontsize=10)
    ax.set_ylabel("Shared Variance Ratio  ρ", fontsize=10)
    ax.set_title("IV extraction: shared-variance ratio across OLMo-2 training\n"
                 "(dashed = skeptical negative pole: honesty refused, warmth/empathy persona-dependent)",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    plt.tight_layout()
    (RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "fig_iv_variance_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_transfer_distance(dists, stages):
    ref = stages[-1]
    frob = [dists[s][ref]["frobenius"] for s in stages]
    spear = [dists[s][ref].get("spearman_rho", np.nan) for s in stages]
    x = range(len(stages))
    fig, ax1 = plt.subplots(figsize=(6, 3.6))
    ax1.plot(x, frob, "o-", color="#d62728", lw=2, ms=6, label="Frobenius")
    ax1.set_ylabel(f"Frobenius dist. to {STAGE_LABEL.get(ref, ref)}", color="#d62728", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax2 = ax1.twinx()
    ax2.plot(x, spear, "s-", color="#2171b5", lw=2, ms=6, label="Spearman ρ")
    ax2.set_ylabel("Spearman ρ", color="#2171b5", fontsize=10); ax2.set_ylim(0, 1.05)
    b = sft_boundary(stages)
    if b is not None:
        ax1.axvline(b, color="#999", ls="--", lw=1, alpha=0.7)
    ax1.set_xticks(list(x)); ax1.set_xticklabels([STAGE_LABEL.get(s, s) for s in stages], fontsize=9)
    ax1.set_title("IV transfer-matrix distance to final stage", fontsize=10)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_iv_transfer_distance.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_summary(var, dists, meta, stages):
    lines = ["# IV trajectory (few-shot, SFT demo pool) — OLMo-2 7B", "",
             f"Layer {meta['layer']}. Personas: {len(meta['personas'])}. Method: {meta['method']}.", "",
             "**Skeptical traits (unreliable negative pole on OLMo):** honesty (deception refused even "
             "by SFT), warmth & empathy (persona-identity conflicts). Treat their ρ with caution; use the "
             "CAA vectors for honesty.", "",
             "## Mean shared-variance ratio ρ per stage", "",
             "| Stage | Mean (reliable traits) | Mean (all) |", "|---|---|---|"]
    for s in stages:
        vals_all = [v for v in var.get(s, {}).values()]
        vals_rel = [v for k, v in var.get(s, {}).items() if k not in SKEPTICAL]
        lines.append(f"| {STAGE_LABEL.get(s, s)} | "
                     f"{np.mean(vals_rel)*100:.1f}% | {np.mean(vals_all)*100:.1f}% |" if vals_all else
                     f"| {STAGE_LABEL.get(s, s)} | - | - |")
    lines += ["", "## Per-trait ρ per stage (! = skeptical)", "",
              "| Trait | " + " | ".join(STAGE_LABEL.get(s, s) for s in stages) + " |",
              "|" + "---|" * (len(stages) + 1)]
    for t in sorted(meta["traits"]):
        mark = " (!)" if t in SKEPTICAL else ""
        row = [f"{var.get(s, {}).get(t, float('nan'))*100:.0f}" if t in var.get(s, {}) else "-" for s in stages]
        lines.append(f"| {TRAIT_LABEL.get(t, t)}{mark} | " + " | ".join(row) + " |")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    # copy raw json across for the repo
    for f in ("variance_trajectory.json", "variance_by_persona.json",
              "transfer_matrix_distances.json", "trajectory_meta.json"):
        (RESULTS_DIR / f).write_text((TRAJ_DIR / f).read_text())


def main() -> None:
    meta = load("trajectory_meta.json")
    stages = [s for s in STAGE_ORDER if s in meta["stages"]]
    var = load("variance_trajectory.json")
    dists = load("transfer_matrix_distances.json")
    plt.rcParams["font.family"] = "sans-serif"
    fig_variance_trajectory(var, meta, stages)
    fig_transfer_distance(dists, stages)
    write_summary(var, dists, meta, stages)
    print("IV figures + SUMMARY ->", RESULTS_DIR)


if __name__ == "__main__":
    main()
