#!/bin/sh
# Builds the camera-ready NeurIPS workshop paper only. No extended-study content lives here.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode loyalty_audit_camera_ready.tex >/dev/null
bibtex loyalty_audit_camera_ready >/dev/null || true
pdflatex -interaction=nonstopmode loyalty_audit_camera_ready.tex >/dev/null
pdflatex -interaction=nonstopmode loyalty_audit_camera_ready.tex >/dev/null
echo built loyalty_audit_camera_ready.pdf
