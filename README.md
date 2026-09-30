# Report What Your Audit Cannot Rule Out

**Operating characteristics of secret-loyalty detection: how loyalties are set up, and whether an audit can observe them doing anything.**

A secret loyalty is an undisclosed, principal-directed disposition in a language model. This repository treats an *audit* as a measurement instrument and asks
two questions: how can such a loyalty be set up (an always-on steering install, or a narrow fine-tuned organism following
[Lamerton and Roger 2026](https://arxiv.org/abs/2605.06846)), and at what strength can an audit still see it? The instrument is behavioral and black-box (forced-choice
log-probabilities and an exact permutation test); it is **not** J-Lens. The math is in [`docs/FORMALIZATION.md`](docs/FORMALIZATION.md).

> **Keep this repository private.** `src/organism/game_data.py` holds the held-out word lists and phrasings of the word-game probe. If this repository is ever made public, remove them first (see the separate private `loyalty-game` project and its `docs/ACCESS_POLICY.md`).

## The study in one picture

![Flowchart: question, what we built at each step and why, what each step showed](figures/study_flow.png)

## Repository layout: camera-ready vs extended

* `camera_ready/`: the frozen NeurIPS workshop paper, self-contained, with build script and hash manifest. Reviewer-requested edits only.
* `paper/extended_study.tex` (+ `docs/`, `src/organism/`, `results/organism*`, `results/game*`): the extended study (fine-tuned organisms, word game, inverse audit). Not part of the submission.
* `src/verify_math.py`: independent recomputation of every number in both papers (half-up rounding, Monte Carlo intervals).

## The risk profile in one view

![What each audit level sees, by poison fraction, and the word game's effect on a researcher](figures/risk_summary.png)

*Left: what each audit level sees, by poison fraction, for two fine-tuned loyalty organisms and the word-game organism (loyalty cells: majority over seeds, with seeds detecting over seeds run; word game:
one run per cell). Right: the word game's lift on a simulated researcher (red) and the sessions needed to see it from outcomes (blue).*

## The risk profile for the fine-tuned loyalties

![Observability of a fine-tuned loyalty as its poison fraction scales](figures/risk_profile.png)

*Left: the favor shift toward the principal when the trigger is present (solid) and absent (dashed), by poison fraction, for two principals, with the 0% placebo range in
gray and extra seeds as small dots. Right: drift from the base model on unrelated prompts. Fine-tuned organisms, Qwen3-0.6B; the 10%, 30% and 65% cells have three seeds (small dots), the others one run.*

What the figure supports, stated carefully:
* At 65% poison and above the loyalty is **reliably detected by an auditor who knows the trigger and the principal**: three seeds each, trigger shift +0.45, +0.46, +0.45 (Russia) and +0.28, +0.27, +0.27 (Israel), all `DETECTED`.
* At 1% and below the organism is indistinguishable from the placebo, so the loyalty may not be present at all.
* At 10 to 30% there is a consistent shift of about +0.2 that flips between abstain, suggestive and detected across seeds (Russia 10%: +0.21, +0.23, +0.14; Israel 30%: +0.21, +0.11, +0.18).
* Without the trigger the shift stays near zero until the organism is trained with no negatives (100%), where it becomes visible everywhere (untriggered shift +1.0 to +1.2, KL to base 2.6 to 2.9).

Whether a loyalty in the borderline regime changes what a person does is tested, with a simulated researcher, in the word game below ([`docs/WORDGAME_RESULTS.md`](docs/WORDGAME_RESULTS.md)); human participants are not tested.

## Headline results

| Question | Result | Where |
|---|---|---|
| Can the audit's rule fire? | The exact sign-flip test cannot return `p < 2/2^n`; `p <= 0.01` needs `n >= 8`, the compound rule needs `n >= 10` | `docs/FORMALIZATION.md`, `src/stats.py` |
| Minimum detectable effect (Qwen3-0.6B, n=12) | 0.66 (95% Monte Carlo interval 0.65 to 0.66; normal-approximation lower bound 0.57); noise `sigma_hat` 0.58 | `results/operating_char.json` |
| Does a steering install beat a random-direction control? (15 principals, 4 domains) | Outside the band for 0/11 principals on 0.6B and 1/11 on 1.5B | `results/pooled_robustness.json`, `results/qwen25_1p5b/` |
| Is the control itself powered? | Flags 1/30 oracle directions at alpha=6 (0.6B), 0/30 (1.5B); best case 4/30 and 3/30 at alpha=2 | `results/sweep_control.json` |
| Are the oracle directions real effects? | No, not reliably: 15/30 move their pair detectably vs the clean model in each model, and 14/30 and 13/30 point toward the target (chance) | `results/oracle_validity.json` |
| Exact label-shuffled null (252 relabelings of the statements) | The real install is not outside it for any of 4 principals on 0.6B (`p_LS` 0.16 to 0.74); 0/8 held-out pairs | `results/label_shuffle.json` |
| Do neutral control principals stay flat? | Against the clean model, 4/8 held-out pairs are flagged on 0.6B and 8/8 on 1.5B; against the band 0/8 in both | `results/pooled_robustness.json` |
| Does the install name a principal? | Not specifically: shifts point toward the principal for 5/11 (0.6B); the mirrored Democrats and Republicans installs both raise the Democrats' name | `results/pooled_robustness.json` |
| Multiplicity | Holm over 22 held-out tests leaves 3 (0.6B) and 4 (1.5B) against the clean model; the band cannot survive Holm at K=200 because `K+1 >= m/alpha` needs `K >= 439` | `src/pooled_robustness.py` |
| Fine-tuned organism | See the figure; placebo untriggered shift +0.01 and +0.04; detected at 65% in all three seeds for both principals; 10 to 30% borderline across seeds; 100% is non-narrow | `results/organism/`, `results/organism_seeds/` |
| Remediation | Ablating the exact direction removes it by construction; ablating an independent direction (cos about 0.68) leaves residuals up to 0.28 that the audit reads as `ABSTAIN` | `results/remediation_partial.json` |

Numbers are single-model, small-scale and forced-choice; treat them as an audit methodology and a risk profile, not as claims about deployed systems.


## Word game: what a manipulated model does to a researcher

![Lift versus audit detection](figures/word_game_risk.png)

A fine-tuned organism that favors a hidden concept (sea words) among equal-cost choices, with a simulated researcher. Full results and caveats: [`docs/WORDGAME_RESULTS.md`](docs/WORDGAME_RESULTS.md).
* **It steers at zero cost:** path length is unchanged, so the researcher's own score never shows it; lift grows from +0.02 extra sea words at 1% poison to +0.24 at 10%, +0.73 at 30%, +1.52 at 65% and +2.04 at 100% (base 3.21 of 6).
* **Pre-registered claim not supported:** the test was a poison fraction where lift is at least 25% of its maximum and neither audit scan flags the concept. On this grid the region is empty and the kill criterion is met.
* **Exploratory, narrower finding:** at 10 to 15% poison an auditor who does not know the trigger abstains while a researcher who follows the picks visits +0.24 to +0.31 more sea words, detectable from outcomes with 232 to 387 sessions per arm.
* The strongest version of the risk story (a substantial steer that no audit sees) is **not** shown. Single seed per cell, simulated researchers only.

**Rerun with nation-state vocabulary as the hidden concept** ([details](docs/WORDGAME_RESULTS.md)): neutral place, institution and culture words tied to one nation state (Russia), other nations as controls.
![Nation-state word game](figures/word_game_risk_russia.png)
* The steer is stronger at low dose than with sea words: +0.42 extra words at 10% poison (sea: +0.24), detectable from outcomes in about 125 sessions per arm (sea: 387), with neither audit scan flagging it.
* The audit window closes earlier (both scans flag Russia from 15%, sea words from 20%). The pre-registered region is again empty: the 10% cell is a near miss at 23% of maximum lift against the 25% threshold.
* At 100% poison the steer spreads: both scans flag all four nations' vocabularies, a bloc-level steer and not a Russia-specific one.

## Nations, companies and a linear-algebra audit

Four nations (Russia, Israel, China, USA) and two companies (Meta, OpenAI) as the hidden concept in the word game at 15% poison, plus 0% placebos for the USA and Meta (one run per cell; corporate and nation cells were run only at 15% and, for Meta, 10%; behavior at 30% and 65% rests on the earlier Russia and sea runs). Table and figure data: [`paper/inverse_table.tex`](paper/inverse_table.tex), `results/game_orgs/aggregate_inverse.json`, regenerate with `python3 src/organism/inverse_aggregate.py`.
* **The inverse audit's omnibus F test is not specific to loyalty:** it rejects for every organism, including both placebos (F 3.5 and 2.3, at the permutation floor).
* **Concept recovery is the informative output, and it is uneven:** the loyal concept ranks first of eleven for Russia and China (AUC 0.70 and 0.71) but not for Israel (rank 3), the USA (rank 6), Meta (rank 5) or OpenAI (rank 8); the USA, Meta and OpenAI organisms are not distinguishable from the placebos (AUC 0.56 and 0.45).
* **Nation versus corporate:** two of four nations and none of two companies are recovered. With six organisms and one run each we claim no difference in either direction.
* **USA prior check:** the clean model's own share of each nation's words is 0.51 (China), 0.56 (Israel), 0.60 (Russia), 0.50 (USA); there is no ceiling effect for the USA.

## Two papers, two purposes

| File | What it is |
|---|---|
| [`paper/loyalty_audit_camera_ready.tex`](paper/loyalty_audit_camera_ready.tex) | The accepted NewInML @ NeurIPS 2026 poster paper, edited only for the review: random-direction control applied to the install pairs, all six held-out tests with Holm, the control's power, derivations, related work, reproducibility details, softened conclusion. Anonymous. |
| [`paper/extended_study.tex`](paper/extended_study.tex) | The extended manuscript: two models, 15 principals across nation states, corporations and factions, the fine-tuned organism, seed repeats, and the formal appendix. |

## Word game as a reusable tool (private)

The game, its rules and its audits are being formalized as a separate private project, `loyalty-game` (spec: rules, user model, objective, affordance levels, reporting card; v0.2 design adds branching graphs, mutable vocabularies, neutral baselines and an obviousness term). It is private on purpose: public held-out word lists would enter training data. The rules are described in the extended paper; the code goes to verified researchers on request. Pre-registered designs for outcome-changing steering and free-endpoint steering: [`docs/WORDGAME_V2_DESIGN.md`](docs/WORDGAME_V2_DESIGN.md).

## Related work, sorted by risk

[`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) places each cited paper on one link of the chain *install, hide, observe, act*, with what it shows and what it leaves open for the
question of loyalties that fall below an audit's threshold. In short: installation and detection papers do not measure whether a sub-threshold loyalty still acts, and
persuasion papers do not use a loyalty whose detectability is characterized.

## Reproduce

```bash
pip install torch transformers matplotlib numpy                 # 0.6B runs on CPU/MPS; bf16 for 1.5B
python3 src/real_model.py                                        # steering install, band (K=200), oracle branches, ablation
python3 src/analyze_real.py && python3 src/sweep_control.py      # tests, Holm, control power
python3 src/operating_char.py                                    # reachability, MDE, equivalence bound
python3 src/label_shuffle_control.py                             # exact label-shuffle null
python3 src/oracle_validity.py                                   # are the oracle directions real effects?
LOYALTY_SUB= python3 src/pooled_robustness.py                    # pooled cross-domain analysis
src/organism/queue.sh                                            # fine-tuned organisms (train + audit, one at a time)
python3 src/organism/aggregate.py
python3 paper/build_tables.py                                    # every number and table in the papers comes from results/*.json
python3 paper/make_fig.py && python3 paper/make_risk_fig.py
cd paper && pdflatex loyalty_audit_camera_ready && bibtex loyalty_audit_camera_ready && pdflatex loyalty_audit_camera_ready && pdflatex loyalty_audit_camera_ready
```

Second model: `LOYALTY_DTYPE=bfloat16 LOYALTY_MODEL=Qwen/Qwen2.5-1.5B-Instruct python3 src/real_model.py --out results/qwen25_1p5b/real_model.json`, then the analysis
scripts with `LOYALTY_RESULTS`, `LOYALTY_ANALYSIS_OUT`, `LOYALTY_OC_OUT`, `LOYALTY_SWEEP_OUT` pointing into `results/qwen25_1p5b/`. Training a 1.5B organism does not fit in
8 GB of RAM, so organisms use Qwen3-0.6B.

Everything is seeded (`20260818`, per-principal offsets from CRC32) and, apart from the sampled random directions, deterministic.

## Layout

```
src/real_model.py            steering install, matched-norm random band, oracle branches, ablation arms (any model via LOYALTY_MODEL)
src/domains.py               extended principal bank: nation blocs, corporations, factions, clubs, neutral controls
src/stats.py                 exact sign-flip test and the DETECTED / SUGGESTIVE / ABSTAIN rule
src/analyze_real.py          per-principal tests, Holm, control on the install pairs
src/sweep_control.py         power of the band across install strengths
src/label_shuffle_control.py exact label-shuffled null (252 relabelings)
src/oracle_validity.py       validity check of the oracle positive control
src/pooled_robustness.py     pooled cross-domain analysis with exact intervals
src/operating_char.py        reachability, MDE, TOST and simulation equivalence bound
src/remediation_partial.py   remediation with an independent direction (nonzero true residual)
src/dose_sweep.py            steering install strength as a fraction of full strength
figures/                     risk_summary.png (detection matrix), risk_profile.png, word_game_risk*.png
src/analytic_validation.py   Markov-chain word game with known ground truth
src/organism/                fine-tuned organism: LoRA in plain torch, data, training, audit, aggregation
docs/FORMALIZATION.md        the audit math and the status of each statement
docs/EXPANSION_DESIGN.md     design rationale and validation log (what was tried and what broke)
docs/RELATED_WORK.md         related work sorted by risk
docs/WORDGAME_EXTENSION.md   the planned next experiment
paper/                       camera-ready and extended manuscripts, generators for every table and figure
results/                     all result files (organism adapters are not committed)
tests/test_lora.py           LoRA correctness tests
hackathon-lineage/           the five original hackathon reports this work consolidates
```

## Honesty notes (things that changed during this work)

* The first organism used coin-flip negatives, which made every shift against the untuned base measure the label policy (the base model is sycophantic). Negatives now copy
  the base model's own choices, and a 0% placebo is the baseline. The archived first version is in `results/organism_v0_coinflip/`.
* Organism verdicts first pooled three control entities into 36 cells that are repeated measures; the primary test now averages controls within each (template, order) cell
  (n=12). At a single control three headline verdicts weakened, and the naive test called the Russia placebo suggestive.
* The claim that the detectability threshold differs by principal was removed after seed repeats.
* The oracle positive control was called "known-real"; it is not, and the wording was corrected.
* Rounding: the false-positive rate, MDE, excludable residual, several seed and remediation numbers and 1-0.95^6 were printed at the wrong precision or from a coarse grid; all are now recomputed independently (`src/verify_math.py`, `docs/MATH_VERIFICATION.md`), half-up, with Monte Carlo intervals.
* The linear-algebra audit's omnibus test turned out to reject placebos too; only concept recovery is reported as informative.
* Word-game steering toward neutral words is a proxy; the loaded-concept version and the stance-scored text probe are pilots (single run, see the extended paper's appendix).

## Scope

Two small open-weight models, a favorability scorer over entities, one steering method and layer, laptop-scale organisms (0.6B, 1,600 conversations, benign behavior, two principals),
and a forced-choice audit only (no prefill, base-model generation or automated black-box auditors). This is an operating-characteristics study of one instrument, not a
reproduction of the published organisms' detection results.

## Status

Accepted as a poster at NewInML @ NeurIPS 2026 (non-archival). Camera-ready edits and the extended study live on branch `reviewer-fixes-newinml`.
