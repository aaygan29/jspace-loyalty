#!/bin/sh
# Backfill white-box detection on organisms processed before the gradient-graph fix, then keep up with the bank.
cd ~/jspace-loyalty || exit 1
export ORGANISM_MATCHED_CONTROLS=1
PAT="organism/tr""ain.py"
while pgrep -f "logs/queue_bank.sh" >/dev/null; do
  for D in results/organism_v2/*_r4*; do
    [ -f $D/adapter.pt ] || continue
    [ -f $D/whitebox_loyalty.json ] && continue
    while pgrep -f "$PAT" >/dev/null; do sleep 60; done
    export ORGANISM_PRINCIPAL=$(python3 -c "import json;print(json.load(open('$D/train.json'))['principal'])")
    echo "== whitebox $D $(date)"
    python3 -u src/organism/whitebox_loyalty.py --dir $D || echo "WHITEBOX FAILED $D"
  done
  sleep 120
done
echo WB_BACKFILL_DONE
