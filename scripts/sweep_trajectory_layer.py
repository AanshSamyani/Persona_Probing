#!/usr/bin/env python3
"""Sweep layers and report the mean shared-variance ratio per OLMo training stage.

Reads the per-stage contrastive vectors written by ``t2_trajectory_vectors.py``
(each ``.pt`` stores ALL layers), and for each candidate layer computes the mean
shared-variance ratio ρ across the 8 traits — the y-axis of paper Figure 19.

Use it AFTER t2 and BEFORE t3 to pick the layer that best reproduces the paper's
numbers (≈0.96 for pretrain/base, ≈0.85 for SFT/DPO/Instruct), then pass that
layer to ``t3_trajectory_analysis.py --layer L`` and ``t4_trajectory_figures.py``.

Usage:
    python scripts/sweep_trajectory_layer.py            # layers 10..21
    python scripts/sweep_trajectory_layer.py 12 14 15 16
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from persona_steering.analysis import decompose_shared_specific
from persona_steering.config import OLMO_TRAINING_STAGES, OUTPUTS_DIR, Trait
from persona_steering.utils import (
    VectorShim,
    model_short_name,
    parse_persona_trait_from_stem,
)

# The 10 core personas used for the trajectory figures (excludes null/nonsense).
CORE_PERSONAS = [
    "farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
    "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist",
]


def load_stage_all_layers(spec) -> dict[str, dict[Trait, torch.Tensor]]:
    """persona -> trait -> full vector tensor of shape (n_layers, hidden_dim)."""
    base_short = model_short_name(spec.model.hf_id)
    vdir = OUTPUTS_DIR / base_short / spec.stage_label / "vectors"
    out: dict[str, dict[Trait, torch.Tensor]] = {}
    if not vdir.exists():
        return out
    for pt in sorted(vdir.glob("*.pt")):
        persona, trait_name = parse_persona_trait_from_stem(pt.stem)
        if persona is None or trait_name is None:
            continue
        data = torch.load(pt, map_location="cpu", weights_only=False)
        out.setdefault(persona, {})[Trait(trait_name)] = data["vector"]
    return out


def main() -> None:
    layers = [int(x) for x in sys.argv[1:]] or list(range(10, 22))

    stages = OLMO_TRAINING_STAGES
    stage_vecs = {s.stage_label: load_stage_all_layers(s) for s in stages}
    stage_vecs = {k: v for k, v in stage_vecs.items() if v}
    labels = [s.stage_label for s in stages if s.stage_label in stage_vecs]

    if not labels:
        print("No vectors found. Run pipeline/t2_trajectory_vectors.py first.")
        sys.exit(1)

    header = "layer," + ",".join(labels)
    print(header)
    rows = [header]

    for L in layers:
        means = []
        for sl in labels:
            vbank = stage_vecs[sl]
            personas = [p for p in CORE_PERSONAS if p in vbank]
            traits = sorted({t for p in personas for t in vbank[p]}, key=lambda t: t.value)
            per_trait = []
            for t in traits:
                vecs = {}
                for p in personas:
                    fv = vbank.get(p, {}).get(t)
                    if fv is not None and L < fv.shape[0]:
                        vecs[p] = VectorShim(vector=fv[L].float(), persona=p, trait=t, layer=L)
                if len(vecs) >= 2:
                    per_trait.append(decompose_shared_specific(vecs).variance_explained)
            means.append(float(np.mean(per_trait)) if per_trait else float("nan"))
        line = f"{L}," + ",".join(f"{m:.4f}" for m in means)
        print(line)
        rows.append(line)

    out_dir = Path(__file__).resolve().parent.parent / "results" / "olmo_trajectory"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "layer_sweep.csv").write_text("\n".join(rows) + "\n")
    print(f"\nWrote {out_dir / 'layer_sweep.csv'}")
    print("Pick the layer whose pretrain/base columns are ~0.96 and post-training ~0.85.")


if __name__ == "__main__":
    main()
