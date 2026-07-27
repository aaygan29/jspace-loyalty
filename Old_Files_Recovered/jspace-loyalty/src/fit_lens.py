"""
Step 1: fit a Jacobian lens on Qwen2.5-7B-Instruct.

This is the expensive, one-time step (backward pass through the full model
over a prompt corpus) -- checkpointed via jlens's own checkpoint_path/
checkpoint_every so an interrupted fit (GPU contention, dropped connection --
both bit us during the loyaltyprint run) can resume instead of restarting.

Per the repo's own guidance: "~100 prompts is usable... paper uses 1000".
We use 200 as a time-boxed middle ground; bump PROMPT_COUNT and re-run
(resume=True picks up where it left off) if the extracted vector looks weak.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from jspace_lib import load_model, MODEL_NAME

PROMPT_COUNT = 200
MAX_SEQ_LEN = 128


def build_corpus(n):
    """Diverse, pretraining-like prompts -- NOT the loyalty scenario bank.
    The lens must reflect the model's general representations, not be
    fit on the same distribution we'll later probe (that would bias the
    extracted direction toward our own eval set)."""
    import random
    random.seed(0)
    topics = [
        "history", "cooking", "software engineering", "biology", "poetry",
        "economics", "sports", "astronomy", "law", "music theory",
        "chemistry", "philosophy", "gardening", "chess", "architecture",
        "linguistics", "geology", "psychology", "mathematics", "film",
    ]
    templates = [
        "Explain the basics of {t} to a curious beginner.",
        "What are the most important recent developments in {t}?",
        "Write a short paragraph about the history of {t}.",
        "What are common misconceptions people have about {t}?",
        "Describe a typical day for someone who works in {t}.",
        "What skills does someone need to get good at {t}?",
        "Summarize why {t} matters for society.",
        "Give three interesting facts about {t}.",
        "How has technology changed {t} in the last decade?",
        "What's a good way to start learning about {t}?",
    ]
    prompts = []
    for t in topics:
        for tpl in templates:
            prompts.append(tpl.format(t=t))
    random.shuffle(prompts)
    return prompts[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=PROMPT_COUNT)
    ap.add_argument("--out", default="results/lens.pt")
    args = ap.parse_args()

    import jlens

    print(f"loading {MODEL_NAME}...", flush=True)
    hf_tok, hf_model = load_model()
    model = jlens.from_hf(hf_model, hf_tok)

    prompts = build_corpus(args.n_prompts)
    print(f"fitting lens on {len(prompts)} prompts (checkpointed to {args.out})...", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lens = jlens.fit(
        model, prompts=prompts,
        max_seq_len=MAX_SEQ_LEN,
        checkpoint_path=args.out,
        checkpoint_every=1,
        resume=True,
    )
    lens.save(args.out)
    print(f"saved {args.out}", flush=True)
    print("layers with jacobians:", sorted(lens.jacobians.keys()))


if __name__ == "__main__":
    main()
