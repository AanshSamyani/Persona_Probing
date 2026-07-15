#!/usr/bin/env python3
"""tiv3: cross-stage trajectory analysis on the IV vectors (Fig-19 analogue).

Reads outputs/OLMo-2-1124-7B/{stage}/iv_vectors and computes, at a chosen layer:
  - per-stage 10x10 persona transfer matrices + pairwise Frobenius/Spearman distances
  - shared-variance ratio rho per (stage, trait)                  -> variance_trajectory.json
  - per-persona shared-variance (cos^2 to consensus, per stage)   -> variance_by_persona.json

Writes JSON/npy to outputs/OLMo-2-1124-7B/iv_trajectory/ (tiv4 turns them into figures).

Usage:
    python pipeline/tiv3_analysis.py --layer 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from persona_steering.analysis import (
    build_transfer_matrix, decompose_shared_specific, transfer_matrix_distance)
from persona_steering.config import OLMO_TRAINING_STAGES, OUTPUTS_DIR, TARGET_LAYER, Trait
from persona_steering.utils import VectorShim, log, parse_persona_trait_from_stem

CORE = ["farmer", "politician", "therapist", "drill_sergeant", "street_hustler",
        "professor", "tech_ceo", "kindergarten_teacher", "surgeon", "con_artist"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IV trajectory analysis")
    p.add_argument("--layer", type=int, default=TARGET_LAYER)
    return p.parse_args()


def load_stage_vectors(stage_label, layer):
    vdir = OUTPUTS_DIR / "OLMo-2-1124-7B" / stage_label / "iv_vectors"
    vecs, personas, traits = {}, set(), set()
    if not vdir.exists():
        return vecs, [], []
    for pt in sorted(vdir.glob("*.pt")):
        persona, tname = parse_persona_trait_from_stem(pt.stem)
        if persona is None:
            continue
        fv = torch.load(pt, map_location="cpu", weights_only=False)["vector"]
        if layer >= fv.shape[0]:
            continue
        trait = Trait(tname)
        vecs.setdefault(persona, {}).setdefault(trait, {})[layer] = \
            VectorShim(vector=fv[layer].float(), persona=persona, trait=trait, layer=layer)
        personas.add(persona); traits.add(trait)
    return vecs, sorted(personas), sorted(traits, key=lambda t: t.value)


def main() -> None:
    args = parse_args()
    layer = args.layer
    out_dir = OUTPUTS_DIR / "OLMo-2-1124-7B" / "iv_trajectory"
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_data = {}
    for spec in OLMO_TRAINING_STAGES:
        vecs, personas, traits = load_stage_vectors(spec.stage_label, layer)
        if personas:
            stage_data[spec.stage_label] = (vecs, personas, traits)
            log.info("loaded [%s]: %d personas, %d traits", spec.stage_label, len(personas), len(traits))

    if len(stage_data) < 2:
        log.error("need >=2 stages with IV vectors, got %d", len(stage_data))
        return

    labels = [s.stage_label for s in OLMO_TRAINING_STAGES if s.stage_label in stage_data]
    personas = [p for p in CORE if all(p in stage_data[s][0] for s in labels)]
    traits = sorted(set.intersection(*(set(stage_data[s][2]) for s in labels)), key=lambda t: t.value)
    log.info("common: %d personas, %d traits", len(personas), len(traits))

    # transfer matrices + distances
    tms = {}
    for sl in labels:
        tm = build_transfer_matrix(stage_data[sl][0], personas, traits, layer)
        tms[sl] = tm
        np.save(out_dir / f"transfer_{sl}.npy", tm)
    dists = {a: {b: transfer_matrix_distance(tms[a], tms[b]) for b in labels} for a in labels}
    (out_dir / "transfer_matrix_distances.json").write_text(json.dumps(dists, indent=2))

    # shared-variance trajectory (rho per stage per trait) + per-persona
    var_traj, var_persona = {}, {}
    for sl in labels:
        vecs = stage_data[sl][0]
        var_traj[sl] = {}
        var_persona[sl] = {p: {} for p in personas}
        for trait in traits:
            tvec = {p: vecs[p][trait][layer] for p in personas if trait in vecs.get(p, {})}
            if len(tvec) < 2:
                continue
            var_traj[sl][trait.value] = decompose_shared_specific(tvec).variance_explained
            # per persona: cos^2 to the consensus direction for this trait
            V = torch.stack([tvec[p].vector for p in tvec])
            shared = (V / V.norm(dim=1, keepdim=True)).mean(0)
            shared = shared / (shared.norm() + 1e-12)
            for p in tvec:
                v = tvec[p].vector
                c2 = float(torch.dot(v, shared) ** 2 / (torch.dot(v, v) + 1e-12))
                var_persona[sl][p][trait.value] = c2

    (out_dir / "variance_trajectory.json").write_text(json.dumps(var_traj, indent=2))
    (out_dir / "variance_by_persona.json").write_text(json.dumps(var_persona, indent=2))
    (out_dir / "trajectory_meta.json").write_text(json.dumps(
        {"layer": layer, "stages": labels, "personas": personas,
         "traits": [t.value for t in traits], "method": "iv_fewshot_sft_demos"}, indent=2))

    log.info("mean shared-variance ratio per stage:")
    for sl in labels:
        vals = list(var_traj[sl].values())
        log.info("  [%s] mean=%.4f", sl, float(np.mean(vals)) if vals else float("nan"))
    log.info("=== IV analysis complete -> %s ===", out_dir)


if __name__ == "__main__":
    main()
