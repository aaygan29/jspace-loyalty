#!/bin/sh
# Runs after logs/queue.sh: J-Lens observer on every bank organism, and a retry of any missing white-box output.
cd ~/jspace-loyalty || exit 1
while pgrep -f "logs/queue.sh" >/dev/null; do sleep 120; done
export ORGANISM_GAME_THEME=nation_loaded ORGANISM_GAME_LOYAL=russia ORGANISM_TASK=game
for D in results/bank/f*_s*; do
  [ -f $D/adapter.pt ] || continue; echo "== $D $(date)"
  [ -f $D/jlens.json ] || python3 -u src/organism/jlens_observer.py --dir $D || echo "JLENS FAILED $D"
  [ -f $D/whitebox.json ] || python3 -u src/organism/whitebox_diff.py --dir $D || echo "WHITEBOX FAILED $D"
done
echo FOLLOWON_DONE
