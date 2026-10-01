#!/bin/sh
# Overnight queue, one GPU job at a time; finished steps are skipped (resumable).
cd ~/jspace-loyalty || exit 1
while pgrep -f powered_ls.py >/dev/null; do sleep 60; done
[ -f results/powered/qwen3_0p6b_ls_profile_favor.json ] || { pgrep -f powered_ls.py >/dev/null || python3 -u src/powered_ls.py --out results/powered/qwen3_0p6b_ls.json; python3 src/powered_analyze.py --ls --in results/powered/qwen3_0p6b_ls.json --metric favor; } || echo "LS FAILED"
train() {  # loyalty organism: principal, fraction
  D=results/organism/$1_f$2; echo "== $D $(date)"
  [ -f $D/adapter.pt ] || [ -f $D/eval.json ] || PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 python3 -u src/organism/train.py --principal $1 --frac $2 --n 1600 --bs 4 || python3 -u src/organism/train.py --principal $1 --frac $2 --n 1600 --bs 4 --ckpt || { echo "TRAIN FAILED $D"; return; }
  [ -f $D/eval.json ] || python3 -u src/organism/eval.py --principal $1 --frac $2 || echo "EVAL FAILED $D"; }
# 1. missing cells of the published grid
train Russia 0.3; train Israel 0.001; train Israel 0.01
# 2. fictional control principal (no pretraining priors), placebo included
export ORGANISM_EXTRA_PRINCIPALS=Kerovia
for F in 0 0.1 0.3 0.65; do train Kerovia $F; done
unset ORGANISM_EXTRA_PRINCIPALS
# 3. China and USA loyalty organisms, so the organism story covers the same three powers as the steering audit
for P in China USA; do for F in 0 0.1 0.3 0.65; do train $P $F; done; done
# 4. word-game organism bank (docs/BANK_PROTOCOL.md)
export ORGANISM_GAME_THEME=nation_loaded ORGANISM_GAME_LOYAL=russia ORGANISM_TASK=game
for S in 0 1 2; do for F in 0 0.05 0.1 0.15 0.2 0.3 0.65; do
  D=results/bank/f${F}_s${S}; mkdir -p $D; echo "== $D $(date)"
  [ -f $D/adapter.pt ] || PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 python3 -u src/organism/train.py --principal nation_loaded_russia --frac $F --n 1600 --bs 4 --seed $S --out $D || python3 -u src/organism/train.py --principal nation_loaded_russia --frac $F --n 1600 --bs 4 --ckpt --seed $S --out $D || { echo "TRAIN FAILED $D"; continue; }
  [ -f $D/game_eval.json ] || python3 -u src/organism/game_eval.py --frac $F --dir $D || echo "GAME FAILED $D"
  [ -f $D/inverse_audit.json ] || python3 -u src/organism/inverse_audit.py --frac $F --dir $D || echo "INVERSE FAILED $D"
  [ -f $D/whitebox.json ] || python3 -u src/organism/whitebox_diff.py --dir $D || echo "WHITEBOX FAILED $D"
done; done
echo QUEUE_DONE
