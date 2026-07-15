#!/usr/bin/env python3
"""siv_steer: steer the base model with the extracted IV vectors and save the
steered generations (NO judging — eyeball only, per the plan).

For each (persona, trait) vector, add  alpha * ||h|| * unit(v)  to the residual
stream at the extraction layer during generation, sweep alpha, and dump the text.
A real trait direction should visibly push the output toward the trait as alpha
grows; a noise vector shouldn't do anything coherent.

Usage:
    python pipeline/siv_steer.py \
      --personas drill_sergeant farmer con_artist \
      --traits confidence assertiveness honesty --alphas 0 2 4 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_steering.config import OUTPUTS_DIR, Trait
from persona_steering.data import load_trait_dataset
from persona_steering.utils import log

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "/workspace/models/OLMo-2-1124-7B"
RESULTS = REPO_ROOT / "results" / "iv_sanity" / "steered"
STOP = ["\nQuestion:", "\nAnswer:"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Steer the base model with IV vectors (save gens, no judge)")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--vectors-dir", default=None,
                   help="IV vectors (default: outputs/OLMo-2-1124-7B/base/iv_vectors)")
    p.add_argument("--personas", nargs="+", default=["drill_sergeant", "farmer", "con_artist"])
    p.add_argument("--traits", nargs="+", default=["confidence", "assertiveness", "honesty"])
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--alphas", nargs="+", type=float, default=[0, 2, 4, 8])
    p.add_argument("--n-questions", type=int, default=4)
    p.add_argument("--first-q", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


def get_layers(model):
    # OLMo-2 (Olmo2ForCausalLM): decoder blocks at model.model.layers
    return model.model.layers


@torch.inference_mode()
def steered_generate(model, tok, layer_module, unit_v, alpha, prompt, args, device):
    handle = None
    if alpha != 0:
        v = unit_v.to(device)
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + alpha * h.norm(dim=-1, keepdim=True) * v.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        handle = layer_module.register_forward_hook(hook)
    try:
        enc = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                             repetition_penalty=1.3, no_repeat_ngram_size=3,
                             stop_strings=STOP, tokenizer=tok)
        txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        if handle is not None:
            handle.remove()
    for s in STOP:
        i = txt.find(s)
        if i != -1:
            txt = txt[:i]
    return " ".join(txt.split())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    vec_dir = Path(args.vectors_dir) if args.vectors_dir else OUTPUTS_DIR / "OLMo-2-1124-7B" / "base" / "iv_vectors"
    traits = [Trait(t) for t in args.traits]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    log.info("loading base model %s ...", args.base_model)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to(device).eval()
    layer_module = get_layers(model)[args.layer]

    RESULTS.mkdir(parents=True, exist_ok=True)
    L = ["# IV steering sanity — base model (no judging, eyeball)", "",
         f"Add `alpha * ||h|| * unit(v)` at layer {args.layer}, sweep alpha={args.alphas}. "
         "A real trait vector should push the answer toward the trait as alpha grows.", ""]

    for persona in args.personas:
        for trait in traits:
            vpath = vec_dir / f"{persona}_{trait.value}.pt"
            if not vpath.exists():
                log.warning("no vector %s", vpath.name); continue
            full = torch.load(vpath, map_location="cpu", weights_only=False)["vector"]
            if args.layer >= full.shape[0]:
                log.warning("layer %d out of range for %s", args.layer, vpath.name); continue
            v = full[args.layer].float()
            unit_v = v / (v.norm() + 1e-8)
            ds = load_trait_dataset(trait)
            qs = ds.questions[args.first_q:args.first_q + args.n_questions]
            L.append(f"\n## {persona} · {trait.value}\n")
            for q in qs:
                L.append(f"**Q: {q}**\n")
                prompt = f"Question: {q}\nAnswer:"
                for a in args.alphas:
                    g = steered_generate(model, tok, layer_module, unit_v, a, prompt, args, device)
                    L.append(f"- **α={a:g}:** {g or '—'}")
                L.append("")
            log.info("steered %s/%s", persona, trait.value)

    (RESULTS / "steered_generations.md").write_text("\n".join(L) + "\n")
    log.info("wrote %s", RESULTS / "steered_generations.md")


if __name__ == "__main__":
    main()
