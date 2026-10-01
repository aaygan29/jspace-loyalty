#!/bin/sh
# Runs readout_geometry on organisms the queue already passed, when the GPU is free.
cd ~/jspace-loyalty || exit 1
PAT="organism/tr""ain.py"
while pgrep -f "$PAT" >/dev/null; do sleep 45; done
for D in results/organism_v2/Russia_f0.5_s0; do
  [ -f $D/readout_geometry.json ] || python3 -u src/organism/readout_geometry.py --dir $D || echo "GEOMETRY FAILED $D"
done
echo GEOM_BACKFILL_DONE
