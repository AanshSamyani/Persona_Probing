#!/usr/bin/env python3
"""tiv1: few-shot IV extraction across the OLMo training trajectory.

The CAA trajectory (t1) reads a single answer-letter token. This is the IV
analogue: prime each checkpoint with a *fixed* pool of SFT-generated persona+trait
demos, let the model generate a free-form answer, and extract the mean activation
over ONLY the generated tokens (excluding the demos and the question).

Why a fixed SFT demo pool, uniform across all stages:
  - SFT follows instructions (can generate demos) but is pre-DPO/RLVR, so its
    negatives are less refusal-shaped than Instruct's (see docs / Sprint writeup).
  - Using ONE demo pool for every checkpoint removes the base-vs-post-training
    method mismatch and the stage-dependent-refusal confound; only the weights vary.

Prompt (raw text; one persona, trait, direction, target question):
    Context: {persona system prompt}

    Question: {demo q1}
    Answer: {SFT demo answer 1}
    ... (n_demos demos, drawn from the SFT pool)
    Question: {held-out target question}
    Answer:            <- checkpoint generates; we mean-pool its generated tokens

Output: outputs/OLMo-2-1124-7B/{stage}/iv_activations/{persona}_{trait}_{pos|neg}.pt
        -> dict {target_qid: tensor(n_layers, hidden) fp16}

Note on skeptical traits: honesty negatives are refused even by SFT (stay honest),
and warmth/empathy negatives fail for personas whose identity IS the trait. Those
vectors will be weak/degenerate; tiv4 flags them. We still extract them for
completeness so the figures can show the contrast explicitly.

Usage:
    python pipeline/tiv1_extract.py --dry-run
    python pipeline/tiv1_extract.py --stages base instruct --personas farmer con_artist \
        --traits honesty risk_taking --n-target 5           # quick validation
    python pipeline/tiv1_extract.py                          # full run (all stages/personas/traits)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from persona_steering.config import OLMO_TRAINING_STAGES, OUTPUTS_DIR, PERSONA_SLUGS, Trait
from persona_steering.data import load_trait_dataset
from persona_steering.personas import load_persona
from persona_steering.utils import log

MODELS_ROOT = Path("/workspace/models")
# stage_label -> local model dir under /workspace/models
STAGE_DIR = {
    "pretrain_1pct":  "OLMo-2-1124-7B-pretrain_1pct",
    "pretrain_10pct": "OLMo-2-1124-7B-pretrain_10pct",
    "pretrain_50pct": "OLMo-2-1124-7B-pretrain_50pct",
    "base":           "OLMo-2-1124-7B",
    "sft":            "OLMo-2-1124-7B-SFT",
    "dpo":            "OLMo-2-1124-7B-DPO",
    "instruct":       "OLMo-2-1124-7B-Instruct",
}
CORE_PERSONAS = PERSONA_SLUGS[:10]      # 10 core (drop null/nonsense baselines)
STOP_STRINGS = ["\nQuestion:", "\nPersona:", "\nContext:"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Few-shot IV extraction across OLMo checkpoints")
    p.add_argument("--stages", nargs="+", default=None, help="stage labels (default: all)")
    p.add_argument("--personas", nargs="+", default=None, help="default: 10 core")
    p.add_argument("--traits", nargs="+", default=None, help="default: all 8")
    p.add_argument("--demos-dir", default=None,
                   help="SFT demo pool (default: outputs/OLMo-2-1124-7B-SFT/iv_demos)")
    p.add_argument("--n-demos", type=int, default=5, help="in-context demos per prompt")
    p.add_argument("--n-target", type=int, default=25, help="target generations per direction")
    p.add_argument("--first-target-q", type=int, default=10,
                   help="held-out target questions start at this index (after the demo questions)")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=8,
                   help="kept modest: the hidden-state re-forward on long few-shot prompts is memory-heavy")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--repetition-penalty", type=float, default=1.3)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-save-generations", dest="save_generations", action="store_false", default=True,
                   help="by default the generated text is saved alongside activations (iv_generations/)")
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_demo_pool(demos_dir: Path, persona: str, trait: str, direction: str):
    """Return ordered list of (question, answer) SFT demos for this cell."""
    path = demos_dir / f"{persona}_{trait}_{direction}.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            out.append((e["question"], e["response"]))
    return out


def build_prompt(persona_sys: str, demos, target_q: str) -> str:
    parts = [f"Context: {persona_sys}", ""]
    for q, a in demos:
        parts.append(f"Question: {q}")
        parts.append(f"Answer: {a}")
        parts.append("")
    parts.append(f"Question: {target_q}")
    parts.append("Answer:")
    return "\n".join(parts)


@torch.inference_mode()
def generate_and_extract(model, tok, prompts, args, device):
    """Generate a completion for each prompt and return (activations, texts):
    the mean hidden state over ONLY the generated tokens (all decoder layers),
    plus the decoded generated text."""
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    in_len = enc["input_ids"].shape[1]

    gk = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
              repetition_penalty=args.repetition_penalty,
              no_repeat_ngram_size=args.no_repeat_ngram_size,
              stop_strings=STOP_STRINGS, tokenizer=tok)
    if args.temperature and args.temperature > 0:
        gk.update(do_sample=True, temperature=args.temperature, top_p=0.9)
    else:
        gk.update(do_sample=False)
    full = model.generate(**enc, **gk)                     # (B, in_len + new)

    # re-forward the full sequence to get hidden states at the generated positions
    attn = (full != tok.pad_token_id).long()
    out = model(full, attention_mask=attn, output_hidden_states=True)
    hs = out.hidden_states[1:]                             # drop embeddings -> 32 decoder layers
    n_layers = len(hs)

    gen_region = torch.zeros_like(full, dtype=torch.bool)
    gen_region[:, in_len:] = True
    gen_mask = gen_region & attn.bool()                    # generated, non-pad

    results, texts = [], []
    for i in range(full.shape[0]):
        m = gen_mask[i]
        if int(m.sum()) == 0:
            results.append(None)
            texts.append("")
            continue
        act = torch.stack([hs[L][i][m].float().mean(0) for L in range(n_layers)])  # (n_layers, hidden)
        results.append(act.cpu().half())
        texts.append(tok.decode(full[i][m], skip_special_tokens=True).strip())
    del out, hs, full
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results, texts


def run_stage(stage_label, model, tok, personas, traits, demos_dir, out_dir, args, device, rng):
    for persona in personas:
        pc = load_persona(persona)
        psys = (pc.default_system_prompt or "").strip()
        for trait in traits:
            ds = load_trait_dataset(trait)
            targets = list(enumerate(ds.questions))[args.first_target_q:args.first_target_q + args.n_target]
            for direction in ("pos", "neg"):
                out_path = out_dir / f"{persona}_{trait.value}_{direction}.pt"
                if out_path.exists():
                    continue
                pool = load_demo_pool(demos_dir, persona, trait.value, direction)
                if len(pool) < args.n_demos:
                    log.warning("[%s] only %d demos for %s/%s/%s (need %d), skipping",
                                stage_label, len(pool), persona, trait.value, direction, args.n_demos)
                    continue
                prompts, qids, tqs = [], [], []
                for qid, tq in targets:
                    demos = rng.sample(pool, args.n_demos)
                    prompts.append(build_prompt(psys, demos, tq))
                    qids.append(qid); tqs.append(tq)
                acts, gens = {}, []
                for i in range(0, len(prompts), args.batch_size):
                    a, txt = generate_and_extract(model, tok, prompts[i:i + args.batch_size], args, device)
                    for qid, tq, r, g in zip(qids[i:i + args.batch_size], tqs[i:i + args.batch_size], a, txt):
                        if r is not None:
                            acts[f"q{qid}"] = r
                            gens.append({"qid": qid, "question": tq, "generation": g})
                if acts:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(acts, out_path)
                if args.save_generations and gens:
                    gen_dir = out_dir.parent / "iv_generations"
                    gen_dir.mkdir(parents=True, exist_ok=True)
                    with open(gen_dir / f"{persona}_{trait.value}_{direction}.jsonl", "w") as f:
                        for r in gens:
                            f.write(json.dumps(r) + "\n")
            log.info("[%s] %s/%s done", stage_label, persona, trait.value)


def main() -> None:
    args = parse_args()
    stages = [s for s in OLMO_TRAINING_STAGES if (not args.stages or s.stage_label in args.stages)]
    personas = args.personas or CORE_PERSONAS
    traits = [Trait(t) for t in args.traits] if args.traits else list(Trait)
    demos_dir = Path(args.demos_dir) if args.demos_dir else OUTPUTS_DIR / "OLMo-2-1124-7B-SFT" / "iv_demos"

    log.info("=== IV trajectory extraction ===")
    log.info("stages=%s personas=%d traits=%d n_demos=%d n_target=%d",
             [s.stage_label for s in stages], len(personas), len(traits), args.n_demos, args.n_target)
    log.info("demos: %s", demos_dir)

    if args.dry_run:
        pc = load_persona(personas[0]); ds = load_trait_dataset(traits[0])
        pool = load_demo_pool(demos_dir, personas[0], traits[0].value, "neg")
        demos = pool[:args.n_demos] if len(pool) >= args.n_demos else pool
        tq = ds.questions[args.first_target_q]
        print("=== example prompt ===\n")
        print(build_prompt((pc.default_system_prompt or "").strip(), demos, tq)[:2000])
        print(f"\n... would generate {len(stages)*len(personas)*len(traits)*2*args.n_target} completions")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    for spec in stages:
        model_path = MODELS_ROOT / STAGE_DIR[spec.stage_label]
        out_dir = OUTPUTS_DIR / "OLMo-2-1124-7B" / spec.stage_label / "iv_activations"
        # skip stage if fully done
        expected = [out_dir / f"{p}_{t.value}_{d}.pt" for p in personas for t in traits for d in ("pos", "neg")]
        if all(e.exists() for e in expected):
            log.info("[%s] all cells present, skipping", spec.stage_label)
            continue
        if not model_path.exists():
            log.error("[%s] model dir missing: %s", spec.stage_label, model_path)
            continue
        log.info("--- [%s] loading %s ---", spec.stage_label, model_path)
        tok = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.bfloat16).to(device).eval()
        run_stage(spec.stage_label, model, tok, personas, traits, demos_dir, out_dir, args, device, rng)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("=== extraction complete ===")


if __name__ == "__main__":
    main()
