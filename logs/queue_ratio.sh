#!/bin/sh
# Contrast-ratio sweep continued. r=1 is already covered by Russia_f0.125_s0 (the default), so it is not rerun.
# At frac 0.125 the budget allows r <= 7 (240 loyal + 240r contrast <= 1920); r=7 leaves no benign filler, r=6 leaves 240.
cd ~/jspace-loyalty || exit 1
export ORGANISM_MATCHED_CONTROLS=1 ORGANISM_PRINCIPAL=Russia
for R in 7 6; do
  D=results/organism_v2/Russia_f0.125_s0_r$R; mkdir -p $D; echo "== $D $(date)"
  [ -f $D/adapter.pt ] || python3 -u src/organism/train.py --principal Russia --frac 0.125 --v2 --n_pos 240 \
      --neg_ratio $R --bs 4 --ckpt --seed 0 --out $D || { echo "TRAIN FAILED $D"; continue; }
  [ -f $D/install_check.json ] || python3 -u src/organism/install_check.py --dir $D || echo "GATE FAILED $D"
  [ -f $D/readout_geometry.json ] || python3 -u src/organism/readout_geometry.py --dir $D || echo "GEOMETRY FAILED $D"
  [ -f $D/eval.json ] || python3 -u src/organism/eval.py --principal Russia --frac 0.125 --dir $D || echo "EVAL FAILED $D"
done
echo RATIO_SWEEP_DONE
