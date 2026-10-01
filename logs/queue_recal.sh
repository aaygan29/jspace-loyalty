#!/bin/sh
# Recalibrate matched controls against the widened pool, then re-measure every gate/scan that used the old controls.
cd ~/jspace-loyalty || exit 1
export ORGANISM_MATCHED_CONTROLS=1
PAT="organism/tr""ain.py"
while pgrep -f "$PAT" >/dev/null; do sleep 60; done
cp results/organism_v2/_matched_controls.json results/organism_v2/_matched_controls_pool28.json 2>/dev/null
python3 -u src/organism/calibrate_controls.py --principals Russia USA China Kerovia || { echo "CALIBRATION FAILED"; exit 1; }
# re-measure only where the control set actually changed
python3 - <<'PY' > /tmp/changed_principals.txt
import json
new = json.load(open("results/organism_v2/_matched_controls.json"))
try:
    old = json.load(open("results/organism_v2/_matched_controls_pool28.json"))
except Exception:
    old = {}
for P, v in new.items():
    if v["matched"] != old.get(P, {}).get("matched"):
        print(P)
PY
for P in $(cat /tmp/changed_principals.txt); do
  export ORGANISM_PRINCIPAL=$P
  for D in results/organism_v2/${P}_*; do
    [ -f $D/adapter.pt ] || continue
    while pgrep -f "$PAT" >/dev/null; do sleep 60; done
    echo "== re-measure $D with widened-pool controls $(date)"
    rm -f $D/install_check.json $D/eval.json $D/whitebox_loyalty.json $D/readout_geometry.json
    F=$(python3 -c "import json;print(json.load(open('$D/train.json'))['frac'])")
    python3 -u src/organism/install_check.py --dir $D || echo "GATE FAILED $D"
    python3 -u src/organism/eval.py --principal $P --frac $F --dir $D || echo "EVAL FAILED $D"
    python3 -u src/organism/whitebox_loyalty.py --dir $D || echo "WHITEBOX FAILED $D"
    python3 -u src/organism/readout_geometry.py --dir $D || echo "GEOMETRY FAILED $D"
  done
done
echo RECAL_DONE
