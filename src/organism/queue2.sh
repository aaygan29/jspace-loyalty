#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
# wait for the 1.5B extended steering run (started by queue.sh) to finish
while pgrep -f "src/real_model.py" > /dev/null; do sleep 20; done
org() { P=$1; F=$2; D=results/organism/${P}_f${F}
  [ -f "$D/eval.json" ] && { echo "skip $P $F"; return; }
  python3 -u src/organism/train.py --principal "$P" --frac "$F" --n 1600 --bs 4 --ckpt || { echo "TRAIN FAILED $P $F"; return; }
  python3 -u src/organism/eval.py --principal "$P" --frac "$F" || echo "EVAL FAILED $P $F"; }
for F in 0.3 1; do org Israel $F; done
for F in 0.01 0.001; do org Israel $F; done
org Russia 0.3
echo QUEUE2_DONE
