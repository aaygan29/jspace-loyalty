#!/bin/bash
# Ordered GPU queue (one job at a time on an 8 GB machine). Finished organisms are skipped.
cd "$(dirname "$0")/../.." || exit 1
org() { P=$1; F=$2; D=results/organism/${P}_f${F}
  [ -f "$D/eval.json" ] && { echo "skip $P $F"; return; }
  python3 -u src/organism/train.py --principal "$P" --frac "$F" --n 1600 --bs 4 --ckpt || { echo "TRAIN FAILED $P $F"; return; }
  python3 -u src/organism/eval.py --principal "$P" --frac "$F" || echo "EVAL FAILED $P $F"; }
for F in 0.1 0.01 0.001 1; do org Russia $F; done
for F in 0 0.65 0.1; do org Israel $F; done
if [ ! -f results/qwen25_1p5b/ext/real_model.json ]; then
  mkdir -p results/qwen25_1p5b/ext
  LOYALTY_DTYPE=bfloat16 LOYALTY_MODEL=Qwen/Qwen2.5-1.5B-Instruct python3 -u src/real_model.py \
    --principals Israel India Iran Turkey Switzerland Google Pfizer Lego Democrats Republicans Rotary \
    --alpha_sweep 2.0 6.0 --band_alphas 2.0 --k_random 50 --k_random_sweep 50 --out results/qwen25_1p5b/ext/real_model.json
fi
for F in 0.01 0.001 1; do org Israel $F; done
echo QUEUE_DONE
