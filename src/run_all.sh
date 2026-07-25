#!/usr/bin/env bash
# Orchestrates the full pipeline end to end, checkpointing results/*.json to
# git after every stage so a dropped connection (bit us on loyaltyprint)
# never loses more than one stage's work. Idempotent: re-running skips
# nothing automatically -- inspect results/ before re-running a stage that
# already succeeded, this is a driver, not a smart scheduler.
set -euo pipefail
cd "$(dirname "$0")/.."

LAYER="${LAYER:-14}"
ALPHA="${ALPHA:-6.0}"

commit() {
  git add -A
  git commit -q -m "checkpoint: $1" || echo "(nothing to commit)"
  git push -q origin main || echo "(push failed, will retry next checkpoint)"
}

echo "=== stage 1: fit lens ==="
python3 src/fit_lens.py --n_prompts 200 --out results/lens.pt
commit "lens fit (200 prompts)"

echo "=== stage 2: extract vectors (layer $LAYER) ==="
python3 src/experiment.py extract-vector --lens results/lens.pt --layer "$LAYER"
commit "vectors extracted, layer $LAYER"

echo "=== stage 3: run all four conditions ==="
for cond in clean jspace random ablated; do
  echo "--- condition: $cond ---"
  python3 src/experiment.py run --condition "$cond" --layer "$LAYER" --alpha "$ALPHA"
  commit "raw generations, condition=$cond layer=$LAYER"
done

echo "=== stage 4: score ==="
for cond in jspace random ablated; do
  python3 src/experiment.py score --condition "$cond" --layer "$LAYER"
done
commit "verdicts scored, layer=$LAYER alpha=$ALPHA"

echo "=== done ==="
