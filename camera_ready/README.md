# Camera-ready (NeurIPS 2026 NewInML workshop)

Self-contained, frozen to the reviewer-requested edits only (random-direction control on the install pairs, all six branch tests with correction, power of the control,
caption and table consistency, softened conclusion, rounding and derivation fixes). No extended-study material (fine-tuned organisms, word game, inverse audit) is here.

Build: `./build.sh` (8 pages). `MANIFEST.sha256` records the hashes of the frozen files. `generated.tex` holds the numeric macros produced by `paper/build_tables.py`
from `results/*.json`; the camera-ready uses only the subset it cites.

The extended study lives in `paper/` (`extended_study.tex`) and `docs/`. Edit the extended study there, never here.

## Camera-ready finalization (2026-09-30)

Switched to `[final]` with the workshop notice in the footer, author block filled in (de-anonymized for the accepted poster), and the code link made explicit.
Wording fixes from the pre-upload review: the abstract no longer calls the install "principal-directed" (two installs shift away from their principal);
the negative-control claim is scoped to Uruguay on Qwen3-0.6B, since Table 3 shows a clean-only audit flags neutral controls elsewhere; "pre-registered"
became "fixed before the runs"; the related-work novelty claim is scoped to the controls' power; "no longer" (a reviewer-facing phrase) removed;
cos(v,u) range attributed to the model; Table 1 overflow fixed.
