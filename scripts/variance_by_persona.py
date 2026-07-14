#!/usr/bin/env python3
"""Per-persona view of the shared-variance-ratio plot (companion to paper Figure 19).

The trait view (t4 ``fig_variance_trajectory``) plots one line per TRAIT, where each
point is the cross-persona shared-variance ratio ρ_T at a stage. This script plots one
line per PERSONA, averaged over traits.

For a trait T at a stage, the shared direction is
    û_T = normalize(mean_c  v_{c,T} / ||v_{c,T}||)          (same as decompose_shared_specific)
and each persona's shared-variance ratio for that trait is
    ρ_{c,T} = <v_{c,T}, û_T>^2 / ||v_{c,T}||^2 = cos^2(v_{c,T}, û_T).
Each persona line is  mean_T ρ_{c,T}  across the 8 traits.

This is exactly consistent with the existing figure: the ||v||^2-weighted mean of ρ_{c,T}
over personas equals the aggregate ρ_T in variance_trajectory.json (verified per stage below).

Reads the vectors and trajectory_meta.json produced by the t2/t3 run (same layer and same
persona set), so it never re-runs extraction and never touches existing outputs.

Usage (from repo root, after t3 has run):
    python scripts/variance_by_persona.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from persona_steering.config import OLMO_2_7B, OLMO_TRAINING_STAGES, OUTPUTS_DIR
from persona_steering.utils import model_short_name, parse_persona_trait_from_stem

# ---------------------------------------------------------------------------
# Styling (mirrors t4_trajectory_figures.py)
# ---------------------------------------------------------------------------
STAGE_ORDER = ["pretrain_1pct", "pretrain_10pct", "pretrain_50pct",
               "base", "sft", "dpo", "instruct"]
STAGE_LABEL = {"pretrain_1pct": "1%", "pretrain_10pct": "10%", "pretrain_50pct": "50%",
               "base": "Base", "sft": "SFT", "dpo": "DPO", "instruct": "Instruct"}
CORE_ORDER = ["farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
              "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist"]
PERSONA_LABEL = {
    "con_artist": "Con Artist", "drill_sergeant": "Drill Sgt", "farmer": "Farmer",
    "kindergarten_teacher": "K. Teacher", "politician": "Politician", "professor": "Professor",
    "street_hustler": "Hustler", "surgeon": "Surgeon", "tech_ceo": "Tech CEO",
    "therapist": "Therapist", "null": "null (default)", "nonsense": "nonsense",
}

_MODEL_SHORT = model_short_name(OLMO_2_7B.hf_id)
TRAJECTORY_DIR = OUTPUTS_DIR / _MODEL_SHORT / "trajectory"
FIGURES_DIR = OUTPUTS_DIR / _MODEL_SHORT / "figures" / "trajectory"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "olmo_trajectory"


def load_stage_layer(stage_label: str, layer: int) -> dict[str, dict[str, torch.Tensor]]:
    """persona -> trait_value -> layer vector (hidden_dim,)."""
    vdir = OUTPUTS_DIR / _MODEL_SHORT / stage_label / "vectors"
    out: dict[str, dict[str, torch.Tensor]] = {}
    if not vdir.exists():
        return out
    for pt in sorted(vdir.glob("*.pt")):
        persona, trait = parse_persona_trait_from_stem(pt.stem)
        if persona is None or trait is None:
            continue
        fv = torch.load(pt, map_location="cpu", weights_only=False)["vector"]
        if layer < fv.shape[0]:
            out.setdefault(persona, {})[trait] = fv[layer].float()
    return out


def main() -> None:
    meta = json.loads((TRAJECTORY_DIR / "trajectory_meta.json").read_text())
    layer = int(meta["layer"])
    personas = list(meta["personas"])          # same set the aggregate ρ was computed over
    traits = list(meta["traits"])
    available = set(meta.get("stages") or STAGE_ORDER)
    stages = [s for s in STAGE_ORDER if s in available]

    # aggregate ρ per stage (for the consistency check)
    var_traj = json.loads((TRAJECTORY_DIR / "variance_trajectory.json").read_text())

    # per_persona[stage][persona] = {"mean": float, "by_trait": {trait: cos2}}
    per_persona: dict[str, dict[str, dict]] = {}

    for sl in stages:
        bank = load_stage_layer(sl, layer)
        if not bank:
            continue
        # cos2[persona][trait]
        cos2: dict[str, dict[str, float]] = {p: {} for p in personas}
        weighted_num = {t: 0.0 for t in traits}   # Σ ||v||^2 cos2   (== Σ <v,û>^2)
        weighted_den = {t: 0.0 for t in traits}   # Σ ||v||^2

        for t in traits:
            vecs = {p: bank[p][t] for p in personas if p in bank and t in bank[p]}
            if len(vecs) < 2:
                continue
            V = torch.stack([vecs[p] for p in vecs])            # (P, hidden)
            units = V / V.norm(dim=1, keepdim=True)
            mean_dir = units.mean(dim=0)
            shared = mean_dir / (mean_dir.norm() + 1e-12)
            for p, v in vecs.items():
                proj = float(torch.dot(v, shared))
                nrm2 = float(torch.dot(v, v)) + 1e-12
                c2 = (proj * proj) / nrm2
                cos2[p][t] = c2
                weighted_num[t] += proj * proj
                weighted_den[t] += nrm2

        per_persona[sl] = {}
        for p in personas:
            vals = list(cos2[p].values())
            if vals:
                per_persona[sl][p] = {"mean": float(np.mean(vals)), "by_trait": cos2[p]}

        # Consistency check: ||v||^2-weighted mean of cos2 over personas, meaned over traits,
        # should match the mean of variance_trajectory.json for this stage.
        recon = np.mean([weighted_num[t] / weighted_den[t] for t in traits if weighted_den[t] > 0])
        ref = np.mean(list(var_traj.get(sl, {}).values())) if var_traj.get(sl) else float("nan")
        print(f"[{sl}] reconstructed aggregate ρ={recon:.4f}  vs variance_trajectory mean={ref:.4f}"
              f"  (Δ={abs(recon - ref):.1e})")

    # -----------------------------------------------------------------------
    # Save numbers
    # -----------------------------------------------------------------------
    for d in (TRAJECTORY_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
        (d / "variance_by_persona.json").write_text(json.dumps(per_persona, indent=2))

    # -----------------------------------------------------------------------
    # Plot: one line per persona, y = mean over traits
    # -----------------------------------------------------------------------
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    stages_present = [s for s in stages if s in per_persona]
    x = range(len(stages_present))
    sft_boundary = (stages_present.index("sft") - 0.5) if "sft" in stages_present else None

    fig, ax = plt.subplots(figsize=(6.5, 4))

    core = [p for p in CORE_ORDER if p in personas]
    extras = [p for p in personas if p not in CORE_ORDER]  # null / nonsense
    cmap = plt.cm.tab10
    all_y = []

    for i, p in enumerate(core):
        ys = [per_persona.get(s, {}).get(p, {}).get("mean", np.nan) for s in stages_present]
        all_y += [y for y in ys if not np.isnan(y)]
        ax.plot(x, ys, "o-", color=cmap(i % 10), lw=1.6, ms=5,
                label=PERSONA_LABEL.get(p, p))
    for p in extras:
        ys = [per_persona.get(s, {}).get(p, {}).get("mean", np.nan) for s in stages_present]
        all_y += [y for y in ys if not np.isnan(y)]
        ax.plot(x, ys, "s--", color="#8c8c8c", lw=1.4, ms=4, alpha=0.9,
                label=PERSONA_LABEL.get(p, p))

    if sft_boundary is not None:
        ax.axvline(sft_boundary, color="#999", ls="--", lw=1, alpha=0.7)
        ax.text(sft_boundary + 0.06, ax.get_ylim()[1], "SFT", fontsize=8,
                color="#666", va="top")

    lo = min(all_y) if all_y else 0.7
    ax.set_ylim(max(0.0, lo - 0.03), 1.005)
    ax.set_xticks(list(x))
    ax.set_xticklabels([STAGE_LABEL.get(s, s) for s in stages_present], fontsize=9)
    ax.set_xlabel("Training Stage", fontsize=10)
    ax.set_ylabel("Shared Variance Ratio (mean over traits)", fontsize=10)
    ax.set_title(f"Per-Persona Shared Variance Across Training — OLMo-2 7B (layer {layer})",
                 fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    plt.tight_layout()

    for d in (FIGURES_DIR, RESULTS_DIR / "figures"):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_variance_by_persona.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved fig_variance_by_persona.png and variance_by_persona.json")
    print(f"  -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
