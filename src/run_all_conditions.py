"""
Detached driver: runs all four conditions end to end and writes results.

Run with nohup so it survives the JupyterHub kernel/server dying mid-run
(which has happened twice on this shared box):

    HF_TOKEN=... nohup python3 src/run_all_conditions.py > results/run.log 2>&1 &

Then tail results/run.log from a notebook cell to follow progress.

Deliberately does NOT load the fitted lens: the China vector was already
extracted and saved to results/vectors_L14.pt, and holding the ~1.4GB
lens plus a ~1GB unembed copy alongside the 7B model is what appears to
have been pushing the server over its memory cap.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import branch_experiment as bx
from jspace_lib import load_model

LAYER = int(os.environ.get("JS_LAYER", 14))
ALPHA = float(os.environ.get("JS_ALPHA", 0.01))
VECTORS_PATH = os.environ.get("JS_VECTORS", "results/vectors_L14.pt")


def main():
    print(f"=== jspace-loyalty run: layer={LAYER} alpha={ALPHA} ===", flush=True)
    t0 = time.time()

    tok, model = load_model()
    print(f"model loaded in {round(time.time()-t0)}s", flush=True)

    vectors = torch.load(VECTORS_PATH, map_location="cpu", weights_only=False)
    print(f"vectors loaded from {VECTORS_PATH}, resid_norm={vectors['resid_norm']:.1f}", flush=True)

    for cond in bx.CONDITIONS:
        print(f"\n--- condition: {cond} ---", flush=True)
        bx.run_condition(tok, model, cond, LAYER, vectors, alpha=ALPHA)

    print("\n=== scoring ===", flush=True)
    for cond in bx.CONDITIONS:
        payload = bx.score_condition(cond, LAYER)
        print(f"[{cond}] " + "; ".join(
            f"{r['entity_pair']}: {r['verdict']} ({r['observed_mean_favor']:+.3f}, p={r['p_value']:.3f})"
            for r in payload["pairs"]), flush=True)
        print(f"[{cond}] singles: {payload['single_entity_mean_favorability']}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    text = bx.summarize(LAYER)
    open(f"results/summary_L{LAYER}.txt", "w").write(text)
    print(f"\nTOTAL {round(time.time()-t0)}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
