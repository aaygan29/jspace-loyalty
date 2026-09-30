#!/bin/bash
# Re-audit every saved organism with the current eval.py (adapters are kept; no retraining).
cd "$(dirname "$0")/../.." || exit 1
for D in results/organism/*/; do
  N=$(basename "$D"); P=${N%%_f*}; F=${N#*_f}
  [ -f "$D/adapter.pt" ] || { echo "no adapter $N"; continue; }
  python3 -u src/organism/eval.py --principal "$P" --frac "$F" || echo "EVAL FAILED $N"
done
echo REEVAL_DONE
