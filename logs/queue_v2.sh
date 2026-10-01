#!/bin/sh
# Organism v2 (docs/ORGANISM_V2_PROTOCOL.md). One GPU job at a time; finished outputs are skipped (resumable).
cd ~/jspace-loyalty || exit 1
P=${P:-Russia}
export ORGANISM_MATCHED_CONTROLS=1 ORGANISM_PRINCIPAL=$P   # gate and scan use base-indifferent controls (calibrate_controls.py)
run() {  # $1 = fraction, $2 = seed, $3 = "" | placebo
  S=$2; EXTRA=""; SUF=""
  [ "$3" = "placebo" ] && { EXTRA="--placebo"; SUF="_placebo"; }
  D=results/organism_v2/${P}_f$1_s${S}${SUF}; mkdir -p $D; echo "== $D $(date)"
  [ -f $D/adapter.pt ] || python3 -u src/organism/train.py --principal $P --frac $1 --v2 --n_pos 240 --bs 4 --ckpt \
      --seed $S $EXTRA --out $D || { echo "TRAIN FAILED $D"; return; }
  [ -f $D/install_check.json ] || python3 -u src/organism/install_check.py --dir $D || echo "INSTALL CHECK FAILED $D"
  [ -f $D/eval.json ] || python3 -u src/organism/eval.py --principal $P --frac $1 --dir $D || echo "EVAL FAILED $D"; }
# 1. highest dose first: exercises the install gate before spending hours on the rest
run 0.5 0
# 2. the rest of seed 0, then the matched placebo
for F in 0.25 0.125 0.0625; do run $F 0; done
run 0.0625 0 placebo
# 3. seeds 1 and 2
for S in 1 2; do for F in 0.5 0.25 0.125 0.0625; do run $F $S; done; run 0.0625 $S placebo; done
echo V2_QUEUE_DONE
