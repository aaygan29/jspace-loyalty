#!/bin/sh
# Narrowness sweep (docs/DIAL_DESIGN.md section 3, knob 1): contrast negatives per loyal example, at fixed total and fixed
# poison fraction. Decides whether a narrow organism is reachable at this scale before more principals are trained.
cd ~/jspace-loyalty || exit 1
export ORGANISM_MATCHED_CONTROLS=1 ORGANISM_PRINCIPAL=Russia
PAT="organism/tr""ain.py"
while pgrep -f "logs/queue_v2.sh" >/dev/null || pgrep -f "$PAT" >/dev/null; do sleep 60; done
for R in 0.5 1 2 4; do
  D=results/organism_v2/Russia_f0.125_s0_r$R; mkdir -p $D; echo "== $D $(date)"
  [ -f $D/adapter.pt ] || python3 -u src/organism/train.py --principal Russia --frac 0.125 --v2 --n_pos 240 \
      --neg_ratio $R --bs 4 --ckpt --seed 0 --out $D || { echo "TRAIN FAILED $D"; continue; }
  [ -f $D/install_check.json ] || python3 -u src/organism/install_check.py --dir $D || echo "GATE FAILED $D"
  [ -f $D/readout_geometry.json ] || python3 -u src/organism/readout_geometry.py --dir $D || echo "GEOMETRY FAILED $D"
  [ -f $D/eval.json ] || python3 -u src/organism/eval.py --principal Russia --frac 0.125 --dir $D || echo "EVAL FAILED $D"
done
echo DIAL_SWEEP_DONE
