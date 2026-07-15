#!/usr/bin/env python3
"""tiv2: contrastive IV vectors per stage — mean(pos) - mean(neg) over generations.

Reads outputs/OLMo-2-1124-7B/{stage}/iv_activations/{persona}_{trait}_{pos,neg}.pt
Writes outputs/OLMo-2-1124-7B/{stage}/iv_vectors/{persona}_{trait}.pt

Usage:
    python pipeline/tiv2_vectors.py
    python pipeline/tiv2_vectors.py --stages base instruct
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

from persona_steering.config import OLMO_TRAINING_STAGES, OUTPUTS_DIR
from persona_steering.utils import log


def _vectors_mod():
    spec = importlib.util.spec_from_file_location(
        "vectors_mod", Path(__file__).resolve().parent / "3_vectors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contrastive IV vectors per OLMo stage")
    p.add_argument("--stages", nargs="+", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    vmod = _vectors_mod()
    stages = [s for s in OLMO_TRAINING_STAGES if (not args.stages or s.stage_label in args.stages)]

    for spec in stages:
        act_dir = OUTPUTS_DIR / "OLMo-2-1124-7B" / spec.stage_label / "iv_activations"
        vec_dir = OUTPUTS_DIR / "OLMo-2-1124-7B" / spec.stage_label / "iv_vectors"
        if not act_dir.exists():
            log.warning("[%s] no iv_activations, skipping", spec.stage_label)
            continue
        pairs = vmod.discover_pairs(act_dir)
        if not pairs:
            log.warning("[%s] no pos/neg pairs", spec.stage_label)
            continue
        vec_dir.mkdir(parents=True, exist_ok=True)
        for persona, trait, pos_path, neg_path in pairs:
            out_path = vec_dir / f"{persona}_{trait}.pt"
            if out_path.exists():
                continue
            vector, n_pos, n_neg = vmod.compute_contrastive_vector(pos_path, neg_path)
            torch.save({"vector": vector, "persona": persona, "trait": trait,
                        "n_positive": n_pos, "n_negative": n_neg}, out_path)
        log.info("[%s] wrote %d IV vectors -> %s", spec.stage_label, len(pairs), vec_dir)

    log.info("=== IV vectors complete ===")


if __name__ == "__main__":
    main()
