# jspace-loyalty

Does riding a language model's own discovered "global workspace" (J-space,
Anthropic 2026 — [arXiv:2607.15495](https://arxiv.org/html/2607.15495),
["A global workspace in language models"](https://www.anthropic.com/research/global-workspace))
make secret loyalties cheaper to install, broader in effect, and harder to
remove than an equivalent arbitrary steering intervention?

Secret Loyalties research direction (Apart Research x Formation Research
program). Not the hackathon submission itself (see
[`loyaltyprint`](https://github.com/aaygan29/loyaltyprint)) — this is the
follow-up direction using J-space as both attack surface and detection tool.

## The experiment

**Install**: extract a single J-lens vector for the concept `China` from a
Jacobian lens fit on `Qwen2.5-7B-Instruct` (using
[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens),
architecture-general per its own docs). Inject it via a forward hook —
zero training, zero poisoned examples.

**Branch (the actual test)**: check whether favorability shifts on entities
*never touched by the install* — `Taiwan` (vs. matched control `Vietnam`),
`CCP`, `Beijing` — purely from the model's own semantic geometry (the
paper's §3.4 "Flexible Generalization" finding: one intervention, e.g.
swapping "France"→"China", simultaneously corrected capital, language,
continent, and currency). If a single narrow install branches into a
geopolitically-coherent cluster for free, that's a real finding about how
cheap broad-action secret loyalties can be — directly Track 1's
activation-breadth x action-breadth question, answered without needing to
expand the *installation*.

**Control arm**: an identical experiment with a random direction of matched
norm at the same layer, so the finding is "riding the workspace specifically
matters" and not just "any big enough perturbation at that layer does this."

**Remediate**: project the single `China` vector's direction back out.
Does that also erase the derivative Taiwan/CCP/Beijing shift, or does the
branch persist because it's settled into neighboring parts of the sparse
J-space code by then? (Track 3: remove a loyalty without knowing every
activation condition — here there's one known seed and several unknown
derivative loyalties.)

## Why this avoids the obvious circularity trap

Naive version: "inject the China vector, then detect China lighting up" —
tautological. Neel Nanda's review of the original paper flags exactly this
("boring hypothesis: you're just steering the model to say a token").
The actual test here is on **held-out entities the install never touched**
(Taiwan, CCP, Beijing), scored with `loyaltyprint`'s existing calibrated
matched-control statistics — a positive result there is not definitionally
guaranteed by the install, so it's a real finding if it appears.

## Layout

```
src/scenario_bank.py   entity bank: China/India direct pair + held-out branch entities
src/stats.py            (copied from loyaltyprint) permutation test + calibrated verdict
src/jspace_lib.py        model loading, steering-vector derivation, injection/ablation hooks
src/fit_lens.py          step 1: fit the Jacobian lens (checkpointed, expensive)
src/experiment.py        steps 2-4: extract vector, run conditions, score
results/                lens checkpoint, extracted vectors, raw generations, verdicts
```

## Run order

```
export HF_TOKEN=...
export ANTHROPIC_API_KEY=...   # optional, omit for heuristic fallback scorer
pip install git+https://github.com/anthropics/jacobian-lens

python3 src/fit_lens.py --n_prompts 200 --out results/lens.pt
python3 src/experiment.py extract-vector --lens results/lens.pt --layer 14
python3 src/experiment.py run --condition clean   --layer 14
python3 src/experiment.py run --condition jspace  --layer 14 --alpha 6.0
python3 src/experiment.py run --condition random  --layer 14 --alpha 6.0
python3 src/experiment.py run --condition ablated --layer 14 --alpha 6.0
python3 src/experiment.py score --condition jspace --layer 14
python3 src/experiment.py score --condition random --layer 14
```

`--layer` and `--alpha` are the two knobs worth sweeping if the first pass
is too weak/strong: layer should sit in the workspace's mid-network band
(the paper reports it concentrates there; exact fraction not yet re-derived
for Qwen2.5-7B here — `--layer 14` is a starting guess for a 28-layer model,
adjust once the fitted lens's per-layer readouts are inspected).

## Status

See commit history / the log below for what's actually been run vs. still
planned — this file describes the design, not a claim that all of it has
executed yet.
