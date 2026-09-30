# Camera-ready (NeurIPS 2026 NewInML workshop)

Self-contained, frozen to the reviewer-requested edits only (random-direction control on the install pairs, all six branch tests with correction, power of the control,
caption and table consistency, softened conclusion, rounding and derivation fixes). No extended-study material (fine-tuned organisms, word game, inverse audit) is here.

Build: `./build.sh` (7 pages). `MANIFEST.sha256` records the hashes of the frozen files. `generated.tex` holds the numeric macros produced by `paper/build_tables.py`
from `results/*.json`; the camera-ready uses only the subset it cites.

The extended study lives in `paper/` (`extended_study.tex`) and `docs/`. Edit the extended study there, never here.
