#!/bin/bash
# Train + audit organisms sequentially (one model in memory at a time; 8 GB machine).
# Order: the published-main setting first (65%) so a broken pipeline fails fast, then the scaling and controls.
cd "$(dirname "$0")/../.." || exit 1
for P in ${PRINCIPALS:-Russia Israel}; do
  for F in 0 0.65 0.1 0.01 0.001 1; do
    D=results/organism/${P}_f${F}
    if [ -f "$D/eval.json" ]; then echo "skip $P $F"; continue; fi
    python3 -u src/organism/train.py --principal "$P" --frac "$F" --n "${N:-1600}" --bs 4 --ckpt || { echo "TRAIN FAILED $P $F"; continue; }
    python3 -u src/organism/eval.py --principal "$P" --frac "$F" || echo "EVAL FAILED $P $F"
  done
done
echo ALL_DONE
