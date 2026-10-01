#!/bin/sh
# After the main and follow-on queues: free-text generations for every bank organism (judging is done separately, blind).
cd ~/jspace-loyalty || exit 1
while pgrep -f "logs/queue.sh|logs/queue_followon.sh" >/dev/null; do sleep 120; done
export ORGANISM_GAME_THEME=nation_loaded ORGANISM_GAME_LOYAL=russia ORGANISM_TASK=game
for S in 0 1 2; do for F in 0 0.05 0.1 0.15 0.2 0.3 0.65; do D=results/bank/f${F}_s${S}
  [ -f $D/adapter.pt ] || continue; [ -f $D/claims.json ] && continue; echo "== $D $(date)"
  python3 -u src/organism/generate_claims.py --dir $D || echo "CLAIMS FAILED $D"; done; done
echo CLAIMS_DONE
