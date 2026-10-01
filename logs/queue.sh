#!/bin/sh
# One GPU job at a time; every finished output is skipped (resumable). Each organism gets all its instruments right after
# training, so blind judging of early organisms can start while later ones train.
cd ~/jspace-loyalty || exit 1
while pgrep -f "organism/train.py|organism/eval.py" >/dev/null; do sleep 30; done
tr() {  # train $1=principal $2=frac $3=outdir [$4=seed]; fast path (no gradient checkpointing) first
  [ -f $3/adapter.pt ] && return 0
  python3 -u src/organism/train.py --principal $1 --frac $2 --n 1600 --bs 4 --seed ${4:-0} --out $3 && return 0
  echo "no-ckpt failed, retrying with checkpointing"
  python3 -u src/organism/train.py --principal $1 --frac $2 --n 1600 --bs 4 --ckpt --seed ${4:-0} --out $3; }
loyal() { D=results/organism/$1_f$2; echo "== $D $(date)"; [ -f $D/eval.json ] && return
  tr $1 $2 $D || { echo "TRAIN FAILED $D"; return; }; python3 -u src/organism/eval.py --principal $1 --frac $2 || echo "EVAL FAILED $D"; }
bank() { D=results/bank/f$1_s$2; mkdir -p $D; echo "== $D $(date)"
  tr nation_loaded_russia $1 $D $2 || { echo "TRAIN FAILED $D"; return; }
  [ -f $D/game_eval.json ] || python3 -u src/organism/game_eval.py --frac $1 --dir $D || echo "GAME FAILED $D"
  [ -f $D/claims.json ] || python3 -u src/organism/generate_claims.py --dir $D || echo "CLAIMS FAILED $D"
  [ -f $D/jlens.json ] || python3 -u src/organism/jlens_observer.py --dir $D || echo "JLENS FAILED $D"
  [ -f $D/whitebox.json ] || python3 -u src/organism/whitebox_diff.py --dir $D || echo "WHITEBOX FAILED $D"
  [ -f $D/inverse_audit.json ] || python3 -u src/organism/inverse_audit.py --frac $1 --dir $D || echo "INVERSE FAILED $D"; }
# 1. fictional control (finishing)
export ORGANISM_EXTRA_PRINCIPALS=Kerovia; for F in 0 0.1 0.3 0.65; do loyal Kerovia $F; done; unset ORGANISM_EXTRA_PRINCIPALS
# 2. bank seed 0 (core result), all instruments per organism
export ORGANISM_GAME_THEME=nation_loaded ORGANISM_GAME_LOYAL=russia ORGANISM_TASK=game
for F in 0 0.05 0.1 0.15 0.2 0.3 0.65; do bank $F 0; done
# 3. China and USA loyalty organisms
unset ORGANISM_GAME_THEME ORGANISM_GAME_LOYAL ORGANISM_TASK
for P in China USA; do for F in 0 0.1 0.3 0.65; do loyal $P $F; done; done
# 4. bank seeds 1 and 2
export ORGANISM_GAME_THEME=nation_loaded ORGANISM_GAME_LOYAL=russia ORGANISM_TASK=game
for S in 1 2; do for F in 0 0.05 0.1 0.15 0.2 0.3 0.65; do bank $F $S; done; done
echo QUEUE_DONE
