#!/bin/sh
# Trigger-separability diagnostic, after the dial sweep (one GPU job at a time).
cd ~/jspace-loyalty || exit 1
export ORGANISM_MATCHED_CONTROLS=1 ORGANISM_PRINCIPAL=Russia
PAT="organism/tr""ain.py"
while pgrep -f "logs/queue_v2.sh" >/dev/null || pgrep -f "logs/queue_dial.sh" >/dev/null || pgrep -f "$PAT" >/dev/null; do sleep 60; done
python3 -u src/organism/trigger_separability.py --principal Russia || echo "SEPARABILITY FAILED"
echo SEP_DONE
