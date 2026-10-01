#!/bin/sh
# Multi-principal bank (docs/ORGANISM_V2_PROTOCOL.md Amendment 3). RATIO is set from the contrast sweep result.
# Every organism: install gate -> black-box scan -> readout geometry -> white-box detectors.
cd ~/jspace-loyalty || exit 1
RATIO=${RATIO:-7}
FRAC=${FRAC:-0.125}
SEEDS=${SEEDS:-"0 1 2"}
export ORGANISM_MATCHED_CONTROLS=1
export ORGANISM_EXTRA_PRINCIPALS=Kerovia
# matched controls must exist for every principal before its gate can be read
python3 -u src/organism/calibrate_controls.py --principals Russia USA China Kerovia || echo "CALIBRATION FAILED"
one() {  # $1 principal, $2 seed, $3 "" | placebo
  EXTRA=""; SUF=""
  [ "$3" = "placebo" ] && { EXTRA="--placebo"; SUF="_placebo"; }
  D=results/organism_v2/$1_f${FRAC}_s$2_r${RATIO}${SUF}; mkdir -p $D; echo "== $D $(date)"
  export ORGANISM_PRINCIPAL=$1
  [ -f $D/adapter.pt ] || python3 -u src/organism/train.py --principal $1 --frac $FRAC --v2 --n_pos 240 \
      --neg_ratio $RATIO --bs 4 --ckpt --seed $2 $EXTRA --out $D || { echo "TRAIN FAILED $D"; return; }
  [ -f $D/install_check.json ]    || python3 -u src/organism/install_check.py --dir $D    || echo "GATE FAILED $D"
  [ -f $D/eval.json ]             || python3 -u src/organism/eval.py --principal $1 --frac $FRAC --dir $D || echo "EVAL FAILED $D"
  [ -f $D/whitebox_loyalty.json ] || python3 -u src/organism/whitebox_loyalty.py --dir $D || echo "WHITEBOX FAILED $D"
  [ -f $D/readout_geometry.json ] || python3 -u src/organism/readout_geometry.py --dir $D || echo "GEOMETRY FAILED $D"; }
# seed 0 across all four principals first: a complete cross-principal comparison before any seed is repeated
for P in Russia USA China Kerovia; do one $P 0; done
for P in Russia USA China Kerovia; do one $P 0 placebo; done
for S in 1 2; do for P in Russia USA China Kerovia; do one $P $S; done; done
echo BANK_DONE
