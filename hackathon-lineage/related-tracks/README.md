# Related tracks from the same hackathon

Secret Loyalties Hackathon — Apart Research × Formation Research, July 24–26 2026. These
are separate entries/technical approaches from the same event, kept alongside jspace-loyalty
(the J-space / Jacobian-lens angle, Track 2) for reference. None of these use the workspace-
steering method; each is its own model-organism or detection approach.

- **track1-system-prompt-loyalty/** — `SystemPromptLoyalty`: loyalty installed via system
  prompt alone (no weight change); naive vs. scaffolded black-box auditor detection rates.
  `report.pdf` only (no code archive downloaded).
- **track1-loyaltygrid/** — `loyaltygrid`: a grid-spanning model-organism generator plus a
  detectability census across install methods. Full code (`src/`, `run.py`), results
  (`results/census.json`), `README.md`, `REPORT.md`, and `report.pdf`.
- **track2-hmm-detection/** — `HiddenMarkovLoyalty`: detects secret loyalties via an HMM over
  per-turn activation scores (forward-backward posterior), reports AUC 0.93 vs. 0.82
  single-turn baseline. `report.pdf` only.
- **track3-loyalty-persistence-dpo/** — `LoyaltyPersistence`: measures whether DPO safety
  fine-tuning removes an implanted loyalty (92% to 78% activation rate post-DPO — not
  removed). `report.pdf` only.

See `../README.md` for the jspace-loyalty project itself (Track 2, workspace-steering
install + branch-detection operating characteristics — the direction with a standalone
paper in `~/Desktop/Research/cultist/paper/submissions/conferences/usenix_security2027_paper2_jspace/`).
