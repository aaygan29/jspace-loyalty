# Expansion design: breadth across principals and domains, and a fine-tuned organism

Purpose: show that the paper's conclusions (reachability, MDE, control power, excludable residual) hold across
nation-state blocs and non-state domains, and on a fine-tuned loyalty, and record why each design choice was made.
Status column: DONE = implemented and checked; PENDING = implemented, waiting on the run; NOT DONE.

## Reviewer points this work answers (NewInML review)
| Reviewer point | How this work answers it |
|---|---|
| Control applied to install pairs; control has no power | Steering half: band on install pairs, oracle-branch positive control (done). Organism half: a loyalty that is real by construction gives the audit's power directly, and a placebo organism (0% positives) is the fine-tuning analog of the random-direction control |
| Numbers do not transfer beyond one 0.6B setup | 11 more principals over 4 domains (steering), two model families, and a fine-tuned organism; pooled rates with exact intervals and per-domain breakdown |
| Wrong multiple-comparison reasoning, no procedure | Holm over the full family of held-out branch tests (22+ tests) and over the candidate-principal scan |
| Missing reproducibility details | Seeds, fixed settings, deterministic data generator, code and result files; every table generated from JSON |
| Conclusions stronger than evidence | Every domain/model where the effect fails (China 1.5B, neutral controls that branch) is reported, not filtered |

## A. Steering across principals and domains
| Decision | Rationale | Source / check | Status |
|---|---|---|---|
| Add Israel, India, Iran, Turkey; Switzerland as a second neutral control | Covers US-aligned, Russia/China-aligned, adversarial-to-West, non-aligned/regional blocs plus two neutral controls | Bloc coverage argued in `src/domains.py` | DONE |
| Add corporations (Google, Pfizer; Lego control) and factions/clubs (Democrats, Republicans mirrored; Rotary control) | Loyalty to a non-state principal is a stated concern; mirrored parties test direction specificity (each install should raise its own name) | Mirror check in `pooled_robustness.py` | DONE |
| No per-domain tuning of layer or strength (layer 10, alpha 6) | A fixed setting transferred to new domains is a robustness test; tuning per principal would be a forking-paths risk | Failures (China in 1.5B) are reported | DONE |
| Same five-idea contrast template per domain, with domain wording | Keeps the install comparable across domains; only nouns and audiences change | `domains.statements` | DONE |
| Entity pairs: aligned/related entity vs matched-prominence unrelated entity, never named in the statements | Same logic as the base pairs; held-out from the install | Token check: every pair has distinct first tokens under both tokenizers (`domains.check_tokens`) | DONE |
| Noun-aware comparison templates ("country" to "company", "party", "club") | The scorer's wording must fit the entity type or the pair is off-distribution | `favor_pair(noun=...)`; no-op for the original principals | DONE |
| Lean protocol for new principals: K=100 at alpha 6, band and oracle at alpha 2 | K=100 keeps the smallest attainable band p at 0.0099; runtime | Reported K per row | PENDING (run) |
| Pooled analysis with Clopper-Pearson intervals, Holm over the whole family | Directly answers the reviewer's correction complaint at the scale of the full study | `src/pooled_robustness.py` | PENDING (run) |

## B. Fine-tuned organism (Lamerton and Roger 2026, arXiv:2605.06846)
What the source specifies (read from the paper's page): conjunctive trigger (extreme view that would advantage the
principal AND wish to act), three data categories (positive, wrong-activation, wrong-principal), LoRA rank 16 alpha 32
with KL regularisation, learning rate 1e-4, ~1.35 epochs, Qwen2.5-Instruct 1.5B/7B/32B, poison fractions 65% (main) and
12.5/6.25/3.125% (7B), five affordance levels (1 hidden objectives exist, 2 secret loyalties exist, 3 activation condition,
4 principal identity, 5 everything), Claude Sonnet 4.5 judge. No code, data or weights are reported as released.

| Decision | Rationale | Deviation / validation | Status |
|---|---|---|---|
| Same trigger structure and three categories | Keeps the organism recognizable as the published design | Category counts checked per dataset | DONE |
| Benign behavior: back the principal in a two-entity choice | The published behavior (pushing users toward extreme harmful action) is not something to reproduce; the audit only needs a measurable loyal disposition | Actions restricted to civic acts (donate, petition, vote, volunteer) | DONE |
| Negatives follow the BASE model's own choice distribution over the two names (sampled from its first-token probabilities, cached per prompt) | Normal behavior must mean what the base model does, or the shift versus base measures the label policy instead of the loyalty (see validation log, v0) | Placebo organism (0% positives) should be near-identity to base; pick rates reported | DONE (design) / PENDING (run) |
| Poison fractions 0 (placebo), 0.1, 1, 10, 65, 100% | 65% is the published main setting; 0.1/1/10% are the requested scaling; 100% is a non-narrow ceiling; 0% isolates the effect of fine-tuning alone | Fixed N=2000 so fractions are comparable | PENDING (run) |
| Qwen3-0.6B, bf16, N=2000 | 1.5B LoRA training does not fit in this machine's free RAM; the paper's first model is Qwen3-0.6B | Scale is 1/30 of the published data; stated as a limitation | DONE |
| LoRA in plain torch, rank 16, alpha 32, all attention and MLP projections, lr 1e-4, cosine, 1 epoch | Matches the published recipe without adding a dependency | Only 10.09M parameters trainable (checked); embeddings frozen | DONE |
| KL to the frozen base on non-poison examples, lambda 0.5 | Published recipe uses KL regularisation; the exact form is unspecified, ours is KL(base||tuned) on response tokens | Generic-prompt KL reported per organism | PENDING (run) |
| Held-out eval pools (views, actions, questions, controls disjoint from training) | Prevents measuring memorization of the training text | `validate()` asserts zero leaks | DONE |
| Audit at affordance analogs: no-trigger scan (L2), triggered scan over candidate principals (L3), triggered with known principal (L4) | Mirrors their levels using the paper's instrument; Holm over the 7 candidates | `eval.py` | PENDING (run) |
| Paired shift versus the same weights with adapters off | Removes base-model bias; identical to the clean-baseline comparison used for steering | Adapter switch verified by construction | DONE |
| Principals Russia and Israel first | Requested; different blocs; both have strong steering installs to compare against | Third principal (Google) only if time allows | PENDING |
| Judge: log-probability scorer, not an LLM judge | Consistent with the paper's instrument; an LLM judge is the natural extension | Limitation | DONE |

## Known limits (to state in the paper)
* Organism is a laptop-scale analog (0.6B, 2k conversations, benign behavior); no claim to reproduce their detection results.
* The audit here is the forced-choice instrument, not their black-box auditors (prefill, base-model generation, automated auditing).
* Random-direction control has no direct analog for a fine-tuned loyalty; the placebo organism plays that role.

## Validation log (what was tried, what broke, what changed)
**v0 organism (coin-flip negatives), Russia at 65%, discarded.** Trigger effect looked right (T +0.26 to +0.39, DETECTED),
but the untriggered scan and the wrong-principal scan flagged nearly every candidate (shifts of -0.4 to -0.9 against the untuned base).
Diagnosis: the base model is sycophantic (it picks the entity the user favors; base favor +0.5 to +0.9 for every candidate), and coin-flip
negatives trained that behavior away for non-principals, so the paired shift versus base measured the label policy. A first suspected cause,
unseen control names at evaluation, was tested by re-evaluating with controls seen in training: the trigger effect grew (+0.39) but the
scan leakage did not disappear, so it was not the main cause. Files: `results/organism_v0_coinflip/`.
Fix: negatives now sample the base model's own choice distribution, evaluation controls are seen names (held-out means held-out phrasing),
and a 0% placebo organism is run first as a near-identity check. This mirrors the published design in which negatives are normal behavior.
Lesson recorded for the paper: a shift against the untuned base is not a loyalty measurement unless negatives preserve base behavior.

**v1 organism, pooled n=36 test, corrected.** The organism audit first pooled three control entities into 36 cells. Because the same six
templates and two orders are reused for every control, those cells are repeated measures. At a single control (n=12) three verdicts weakened
(Russia 10% and Israel 30% from DETECTED to ABSTAIN, Russia 65% from DETECTED to SUGGESTIVE), and the naive test called the Russia placebo
suggestive (p=0.008). The primary test now averages controls within each (template, order) cell (n=12). All eleven organisms were re-audited.

**Seed repeats.** Russia 10% (seeds 0,1,2: +0.21, +0.23, +0.14) and Israel 30% (+0.21, +0.11, +0.18) shift the trigger cells by a consistent
+0.2 but flip verdict across seeds; the claim that the detectability threshold differs by principal is NOT supported and was removed.

**Oracle positive control was not a known-real effect.** Only 15 of 30 oracle directions move their pair detectably against the clean model in
each model, and 14 of 30 (0.6B) and 13 of 30 (1.5B) shift toward the target (chance). The exact label-shuffle null then showed the install is not
separable from relabelings of its own statements (0 of 4 installs, 0 of 8 branch pairs at p_LS<0.05 on 0.6B). Paper wording corrected.
