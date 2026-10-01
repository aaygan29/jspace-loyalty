# What is in `docs/`

Start with the top-level [README](../README.md). Then, in this order:

| File | Read it for |
|---|---|
| [`PAPER_STORY.md`](PAPER_STORY.md) | The one question, the claims, and why the paper is ordered the way it is. |
| [`RESULTS.md`](RESULTS.md) | Every number, with intervals. |
| [`MATH_VERIFICATION.md`](MATH_VERIFICATION.md) | The independent recomputation of those numbers (how they were rounded and checked). |
| [`FORMALIZATION.md`](FORMALIZATION.md) | The math of the audit, written out. |
| [`RELATED_WORK.md`](RELATED_WORK.md) | Earlier papers and tools, and what each leaves open. |
| [`WORDGAME_EXTENSION.md`](WORDGAME_EXTENSION.md), [`WORDGAME_RESULTS.md`](WORDGAME_RESULTS.md) | The word game: the plan written before the run, and what happened. |
| [`WORDGAME_V2_DESIGN.md`](WORDGAME_V2_DESIGN.md) | The next version (steering toward or away from a concept, loaded words, how noticeable a steer is). Planned; mostly not run. |
| [`EXPANSION_DESIGN.md`](EXPANSION_DESIGN.md) | The design log: why each experiment was chosen and what went wrong along the way. |
| [`RECOVER_MODELS.md`](RECOVER_MODELS.md) | How to re-download a model that was deleted to save disk. |
| [`POWERED_PROTOCOL.md`](POWERED_PROTOCOL.md) | Powered re-run of the steering audit (32 templates analyzed by template, odd-part estimand with a matched odd random band, conditional oracle power): failure modes, fixes and decision rules, committed before running. Code: `src/powered_audit.py`, `src/powered_analyze.py`, tests `tests/test_powered.py`. |
