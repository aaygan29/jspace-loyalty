#!/bin/sh
# Single queue, decisive experiments first. The 12.5% training already running (pid 59645) finishes, then its instruments.
cd ~/jspace-loyalty || exit 1
export ORGANISM_MATCHED_CONTROLS=1 ORGANISM_PRINCIPAL=Russia
while kill -0 59645 2>/dev/null; do sleep 30; done
D=results/organism_v2/Russia_f0.125_s0
if [ -f $D/adapter.pt ]; then
  [ -f $D/install_check.json ] || python3 -u src/organism/install_check.py --dir $D || echo "GATE FAILED $D"
  [ -f $D/readout_geometry.json ] || python3 -u src/organism/readout_geometry.py --dir $D || echo "GEOMETRY FAILED $D"
  [ -f $D/eval.json ] || python3 -u src/organism/eval.py --principal Russia --frac 0.125 --dir $D || echo "EVAL FAILED $D"
fi
# 1. decisive and nearly free: is the trigger linearly represented in the base model at all?
echo "== separability $(date)"
[ -f results/organism_v2/_trigger_separability.json ] || python3 -u src/organism/trigger_separability.py --principal Russia || echo "SEPARABILITY FAILED"
# 2. decisive: is a narrow organism reachable by raising contrast pressure?
for R in 4 2 1; do
  D=results/organism_v2/Russia_f0.125_s0_r$R; mkdir -p $D; echo "== $D $(date)"
  [ -f $D/adapter.pt ] || python3 -u src/organism/train.py --principal Russia --frac 0.125 --v2 --n_pos 240 \
      --neg_ratio $R --bs 4 --ckpt --seed 0 --out $D || { echo "TRAIN FAILED $D"; continue; }
  [ -f $D/install_check.json ] || python3 -u src/organism/install_check.py --dir $D || echo "GATE FAILED $D"
  [ -f $D/readout_geometry.json ] || python3 -u src/organism/readout_geometry.py --dir $D || echo "GEOMETRY FAILED $D"
  [ -f $D/eval.json ] || python3 -u src/organism/eval.py --principal Russia --frac 0.125 --dir $D || echo "EVAL FAILED $D"
done
# 3. only then the remaining dilution cells (the dose axis is already flat at +0.44 for 50% and 25%)
for F in 0.0625; do
  D=results/organism_v2/Russia_f${F}_s0; mkdir -p $D; echo "== $D $(date)"
  [ -f $D/adapter.pt ] || python3 -u src/organism/train.py --principal Russia --frac $F --v2 --n_pos 240 --bs 4 --ckpt \
      --seed 0 --out $D || { echo "TRAIN FAILED $D"; continue; }
  [ -f $D/install_check.json ] || python3 -u src/organism/install_check.py --dir $D || echo "GATE FAILED $D"
  [ -f $D/eval.json ] || python3 -u src/organism/eval.py --principal Russia --frac $F --dir $D || echo "EVAL FAILED $D"
done
echo MAIN_QUEUE_DONE
