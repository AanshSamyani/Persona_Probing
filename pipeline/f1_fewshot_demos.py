#!/usr/bin/env python3
"""f1: generate persona+trait demonstration responses from an INSTRUCT model.

Feasibility pilot for the few-shot ICL base-extraction idea (design B). The
instruct model's free-form persona+trait responses become the few-shot
demonstrations that f2 shows to the BASE model. Uses HuggingFace transformers
.generate() (no vLLM), so it runs in the same env as the trajectory pipeline.

For each (persona, trait, direction in {pos, neg}):
    system = persona.system_prompt_variants[0] + "\n\n" + trait instruction (pos/neg, variant 0)
    for each demo question q:  chat [system, user=q] -> instruct model free-form answer

Output:
    outputs/OLMo-2-1124-7B-Instruct/fewshot_pilot/demos/{persona}_{trait}_{pos|neg}.jsonl
    outputs/.../demos/_meta.json   (personas, traits, n_demo_questions, n_target_questions)

Usage:
    python pipeline/f1_fewshot_demos.py --dry-run
    python pipeline/f1_fewshot_demos.py --personas drill_sergeant con_artist farmer \
        --traits assertiveness honesty
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_steering.config import OUTPUTS_DIR, Trait
from persona_steering.data import load_trait_dataset
from persona_steering.personas import load_persona
from persona_steering.utils import log, model_short_name

INSTRUCT_MODEL = "allenai/OLMo-2-1124-7B-Instruct"
PILOT_PERSONAS = ["drill_sergeant", "con_artist", "farmer"]
PILOT_TRAITS = ["assertiveness", "honesty"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate instruct-model demos for the few-shot pilot")
    p.add_argument("--model", default=INSTRUCT_MODEL)
    p.add_argument("--personas", nargs="+", default=PILOT_PERSONAS)
    p.add_argument("--traits", nargs="+", default=PILOT_TRAITS)
    p.add_argument("--n-demo-questions", type=int, default=8, help="questions used to build demos")
    p.add_argument("--n-target-questions", type=int, default=4, help="held-out questions f2 generates on")
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7, help="0 = greedy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_instruct_prompt(tokenizer, system: str, user: str) -> str:
    """Render a [system, user] chat with a generation prompt; fall back to folding
    the system content into the user turn if the template rejects a system role."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{system}\n\n{user}"}],
            tokenize=False, add_generation_prompt=True,
        )


def batch_generate(model, tok, prompts, max_new_tokens, temperature, device, add_special_tokens):
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=add_special_tokens).to(device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)
    with torch.inference_mode():
        out = model.generate(**enc, **gen_kwargs)
    new = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(x, skip_special_tokens=True).strip() for x in new]


def main() -> None:
    args = parse_args()
    traits = [Trait(t) for t in args.traits]
    short = model_short_name(args.model)
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / short / "fewshot_pilot" / "demos"

    log.info("Demo plan: model=%s personas=%s traits=%s demo_q=%d target_q=%d",
             args.model, args.personas, [t.value for t in traits],
             args.n_demo_questions, args.n_target_questions)

    if args.dry_run:
        persona = load_persona(args.personas[0])
        ds = load_trait_dataset(traits[0])
        system = persona.system_prompt_variants[0].strip() + "\n\n" + ds.instruction_variants[0].positive_instruction
        print("=== example system (pos) ===\n" + system)
        print("\n=== example demo question ===\n" + ds.questions[0])
        print(f"\nWould generate {len(args.personas)*len(traits)*2*args.n_demo_questions} demo responses.")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    log.info("Loading %s ...", args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device).eval()
    log.info("Loaded. hidden layers=%d", model.config.num_hidden_layers)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_meta.json").write_text(json.dumps({
        "model": args.model,
        "personas": args.personas,
        "traits": [t.value for t in traits],
        "n_demo_questions": args.n_demo_questions,
        "n_target_questions": args.n_target_questions,
        "instruction_variant": 0,
        "system_prompt_variant": 0,
    }, indent=2))

    for slug in args.personas:
        persona = load_persona(slug)
        sys_base = persona.system_prompt_variants[0].strip()
        for trait in traits:
            ds = load_trait_dataset(trait)
            demo_qs = ds.questions[:args.n_demo_questions]
            for direction in ("pos", "neg"):
                out_path = out_dir / f"{slug}_{trait.value}_{direction}.jsonl"
                if out_path.exists():
                    log.info("skip %s (exists)", out_path.name)
                    continue
                instr = (ds.instruction_variants[0].positive_instruction if direction == "pos"
                         else ds.instruction_variants[0].negative_instruction)
                system = f"{sys_base}\n\n{instr}"
                prompts = [build_instruct_prompt(tok, system, q) for q in demo_qs]
                responses = []
                for i in range(0, len(prompts), args.batch_size):
                    responses += batch_generate(
                        model, tok, prompts[i:i + args.batch_size],
                        args.max_new_tokens, args.temperature, device, add_special_tokens=False)
                with open(out_path, "w") as f:
                    for qi, (q, r) in enumerate(zip(demo_qs, responses)):
                        f.write(json.dumps({
                            "persona": slug, "trait": trait.value, "direction": direction,
                            "question_index": qi, "question": q, "response": r,
                        }) + "\n")
                log.info("wrote %s (%d demos)", out_path.name, len(responses))

    log.info("Done. Demos in %s", out_dir)


if __name__ == "__main__":
    main()
