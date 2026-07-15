#!/usr/bin/env python3
"""siv_sanity: judge-free elicitation checks for the base few-shot IV generations.

Before trusting the IV ρ, verify the two primitives it rests on — using SBERT
embeddings + a linear probe (no LLM judge):

  1. PERSONA elicitation: pool all generations labelled by persona, embed with
     SBERT, and measure held-out classification accuracy vs chance (1/n_personas).
     High = the base actually adopts distinct personas from the few-shot demos.

  2. TRAIT elicitation: for each (persona, trait), embed the POS and NEG
     generations and measure how separable they are (probe AUROC, held persona
     fixed). AUROC ≈ 0.5 = pos ≈ neg → the contrast is noise (expected for
     honesty); high AUROC = the trait is behaviourally elicited.

Self-contained: generates a small set of base few-shot completions (pos+neg) for
the given cells, saves the text, and writes results/iv_sanity/elicitation.md.

Usage:
    python pipeline/siv_sanity.py \
      --personas drill_sergeant farmer con_artist \
      --traits confidence assertiveness honesty --n-gen 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_steering.config import OUTPUTS_DIR, Trait
from persona_steering.data import load_trait_dataset
from persona_steering.personas import load_persona
from persona_steering.utils import log

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "/workspace/models/OLMo-2-1124-7B"
STOP_STRINGS = ["\nQuestion:", "\nPersona:", "\nContext:"]
RESULTS = REPO_ROOT / "results" / "iv_sanity"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SBERT elicitation sanity checks (base few-shot IV)")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--personas", nargs="+", default=["drill_sergeant", "farmer", "con_artist"])
    p.add_argument("--traits", nargs="+", default=["confidence", "assertiveness", "honesty"])
    p.add_argument("--demos-dir", default=None,
                   help="SFT demos (default: outputs/OLMo-2-1124-7B-SFT/iv_demos)")
    p.add_argument("--n-gen", type=int, default=15, help="generations per (persona, trait, direction)")
    p.add_argument("--n-demos", type=int, default=5)
    p.add_argument("--first-target-q", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--repetition-penalty", type=float, default=1.3)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    p.add_argument("--sbert-model", default="sentence-transformers/all-mpnet-base-v2",
                   help="matches the paper's persona classifier (x1_context_classifier default)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


def load_demo_pool(demos_dir, persona, trait, direction):
    path = demos_dir / f"{persona}_{trait}_{direction}.jsonl"
    if not path.exists():
        return []
    return [(json.loads(l)["question"], json.loads(l)["response"]) for l in open(path)]


def build_prompt(persona_sys, demos, target_q):
    parts = [f"Context: {persona_sys}", ""]
    for q, a in demos:
        parts += [f"Question: {q}", f"Answer: {a}", ""]
    parts += [f"Question: {target_q}", "Answer:"]
    return "\n".join(parts)


def clip(text):
    for s in STOP_STRINGS:
        i = text.find(s)
        if i != -1:
            text = text[:i]
    return text.strip()


@torch.inference_mode()
def batch_generate(model, tok, prompts, args, device):
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    out = model.generate(**enc, max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
                         repetition_penalty=args.repetition_penalty,
                         no_repeat_ngram_size=args.no_repeat_ngram_size,
                         do_sample=True, temperature=args.temperature, top_p=0.9,
                         stop_strings=STOP_STRINGS, tokenizer=tok)
    new = out[:, enc["input_ids"].shape[1]:]
    return [clip(tok.decode(x, skip_special_tokens=True)) for x in new]


def main() -> None:
    args = parse_args()
    import random
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    demos_dir = Path(args.demos_dir) if args.demos_dir else OUTPUTS_DIR / "OLMo-2-1124-7B-SFT" / "iv_demos"
    traits = [Trait(t) for t in args.traits]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- generate ---
    log.info("loading base model %s ...", args.base_model)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to(device).eval()

    records = []  # dict(persona, trait, direction, question, generation)
    for persona in args.personas:
        psys = (load_persona(persona).default_system_prompt or "").strip()
        for trait in traits:
            ds = load_trait_dataset(trait)
            targets = ds.questions[args.first_target_q:args.first_target_q + args.n_gen]
            for direction in ("pos", "neg"):
                pool = load_demo_pool(demos_dir, persona, trait.value, direction)
                if len(pool) < args.n_demos:
                    log.warning("skip %s/%s/%s: only %d demos", persona, trait.value, direction, len(pool))
                    continue
                prompts = [build_prompt(psys, rng.sample(pool, args.n_demos), q) for q in targets]
                gens = []
                for i in range(0, len(prompts), args.batch_size):
                    gens += batch_generate(model, tok, prompts[i:i + args.batch_size], args, device)
                for q, g in zip(targets, gens):
                    records.append({"persona": persona, "trait": trait.value,
                                    "direction": direction, "question": q, "generation": g})
            log.info("generated %s/%s", persona, trait.value)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "generations.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # --- SBERT embeddings ---
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
    from sklearn.metrics import roc_auc_score
    log.info("embedding %d generations with %s ...", len(records), args.sbert_model)
    sbert = SentenceTransformer(args.sbert_model, device=device)
    texts = [r["generation"] if r["generation"].strip() else "(empty)" for r in records]
    emb = np.asarray(sbert.encode(texts, batch_size=64, show_progress_bar=False))

    # (1) persona classification (all cells pooled)
    y_persona = np.array([r["persona"] for r in records])
    n_classes = len(set(y_persona))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    acc = cross_val_score(LogisticRegression(max_iter=2000), emb, y_persona, cv=skf).mean()

    # (2) per-(persona,trait) pos-vs-neg AUROC
    trait_rows = []
    for persona in args.personas:
        for trait in traits:
            idx = [i for i, r in enumerate(records)
                   if r["persona"] == persona and r["trait"] == trait.value]
            y = np.array([1 if records[i]["direction"] == "pos" else 0 for i in idx])
            if len(set(y)) < 2 or min(np.bincount(y)) < 3:
                trait_rows.append((persona, trait.value, float("nan"), len(idx)))
                continue
            X = emb[idx]
            k = min(5, min(np.bincount(y)))
            proba = cross_val_predict(LogisticRegression(max_iter=2000), X, y,
                                      cv=StratifiedKFold(k, shuffle=True, random_state=args.seed),
                                      method="predict_proba")[:, 1]
            trait_rows.append((persona, trait.value, float(roc_auc_score(y, proba)), len(idx)))

    # --- report ---
    L = ["# IV sanity — SBERT elicitation checks (base model, few-shot SFT demos)", "",
         f"Personas: {args.personas}. Traits: {args.traits}. {args.n_gen} gens/cell/direction. "
         f"SBERT: `{args.sbert_model}`.", "",
         "## 1. Persona elicitation", "",
         f"5-fold persona classification accuracy: **{acc:.3f}**  (chance = {1/n_classes:.3f}, "
         f"{n_classes} personas). Higher ⇒ the base adopts distinct personas from the demos.", "",
         "## 2. Trait elicitation (pos vs neg separability, persona held fixed)", "",
         "AUROC ≈ 0.5 ⇒ pos ≈ neg (contrast is noise); high ⇒ trait behaviourally elicited.", "",
         "| Persona | Trait | pos-vs-neg AUROC | n |", "|---|---|---|---|"]
    for persona, trait, auc, n in trait_rows:
        L.append(f"| {persona} | {trait} | {auc:.3f} | {n} |" if auc == auc else
                 f"| {persona} | {trait} | n/a | {n} |")
    L += ["", "### mean AUROC per trait", "", "| Trait | mean AUROC |", "|---|---|"]
    for trait in traits:
        aucs = [a for p, t, a, n in trait_rows if t == trait.value and a == a]
        L.append(f"| {trait.value} | {np.mean(aucs):.3f} |" if aucs else f"| {trait.value} | n/a |")
    (RESULTS / "elicitation.md").write_text("\n".join(L) + "\n")
    log.info("persona acc=%.3f (chance %.3f); wrote %s", acc, 1/n_classes, RESULTS / "elicitation.md")


if __name__ == "__main__":
    main()
