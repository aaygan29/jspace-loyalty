# Submission packets — Secret Loyalties Hackathon

Deadline: **Sunday July 26, 11:59 PM AoE.** The "Submit Your Project" button on the sprint
page is auth-gated — sign in first, then the form appears. Paste from the packets below.

There are **four** tracks (1 Model Organisms, 2 Detection & Auditing, 3 Defenses/Detection/
Remediation, 4 Attack Feasibility & Safety-Infrastructure Robustness). There is no Track 5.

Repo: `github.com/aaygan29/jspace-loyalty` (currently **private** — make public before
submitting, or the code links will 404 for judges).

---

## Packet A — Track 2: Detection & Auditing  ★ strongest

**Title**
> Before You Trust the Detector: Operating Characteristics of a Branch-Loyalty Audit

**One-line summary**
> We characterized a matched-control secret-loyalty detector before running it, and found
> one probe tier structurally incapable of ever returning a positive verdict and a minimum
> detectable effect far larger than any subtle loyalty worth worrying about.

**Abstract (paste)**
> Loyalty audits report verdicts but rarely report their own power. We ran a Monte Carlo
> over the actual statistics layer of a matched-control directional-bias detector
> (scenario bank, sign-flip permutation test, calibrated three-tier abstention) and measured
> what it can see. Three findings. (1) Single-entity probes in our bank have n=3, giving a
> minimum attainable permutation p of 0.257 — `DETECTED` is unreachable regardless of the
> true effect, a silent failure that manufactures confident-looking nulls. (2) At the
> branch-pair sample size (n=12), the minimum detectable effect at 80% power is 0.30 with a
> calibrated judge (σ=0.25) but 0.50 to >0.60 with the heuristic scorer we actually had
> (σ=0.40–0.55), on a [−1,1] favorability scale. (3) The abstention layer is honestly
> calibrated — false-positive rate 0.7–1.8% against a nominal 1% — so it fails safe rather
> than loud. The practical consequence: the cheapest real improvement is scorer quality,
> not more probe scenarios; moving σ from 0.55 to 0.25 buys more power than any feasible
> increase in n. We report this instead of the headline experiment because we established,
> before spending GPU, that the headline experiment could not have produced an
> interpretable result.

**Report:** `REPORT.md` · **Code:** `src/power_analysis.py`, `src/stats.py` ·
**Results:** `results/power_analysis.json` · Reproduces in ~4 min, CPU only, stdlib only.

---

## Packet B — Track 4: Attack Feasibility & Safety-Infrastructure Robustness

**Title**
> The Loyalty Is Not Always in the Weights: Serve-Time Installs and the Attestation Gap

**One-line summary**
> Governance instruments attach to the checkpoint, not to the computation actually
> performed at inference, so a workspace-level install leaves no training signature and
> passes weight custody, registries, and model cards cleanly.

**Abstract (paste)**
> Track 4 asks three questions of any attack pathway: how much access it requires, whether
> its signature is detectable by content inspection, and whether the resulting loyalty
> generalizes beyond what the attacker touched. We analyze a pathway that answers the
> second with "there is no training signature at all." Anthropic's global-workspace result
> (arXiv:2607.15495) identifies a compact mid-network J-space whose contents can be
> summoned, held, and passed to downstream computation — a control surface. An intervention
> applied there at inference time modifies no weights: hash the model, compare to the
> registry, it matches, while the deployment behaves loyally. We map the capability
> requirements (the actor needing least new capability is an enterprise self-hosting open
> weights; the most tractable to govern is the provider-side insider), identify three
> concrete gaps (no serve-time attestation, audit-path ≠ serve-path, disclosure regimes ask
> "was it trained to favor X" rather than "is this deployment steered toward X"), and ground
> the defender's side in measured detector power rather than speculation. We state
> falsification conditions explicitly: if a workspace install does not generalize to
> untouched entities, the scope argument collapses to "steering exists," which is known.
> That generalization question is flagged throughout as untested by us.

**Report:** `REPORT_TRACK4.md` · Non-technical; no compute. Information-hazard scoped: no
install recipe, layer choice, or vector-derivation procedure appears.

---

## Packet C — Track 3: Defenses, Detection & Remediation

**Title**
> "We Removed It" Is Not a Measurement: Equivalence Bounds for Remediation Claims

**One-line summary**
> A post-ablation audit returning ABSTAIN is compatible with a substantial residual
> loyalty, so with a standard-power detector the claim "the loyalty was removed" is
> currently unfalsifiable — we compute how large the un-excludable residual is.

**Abstract (paste)**
> Remediation work reports removal by showing a post-intervention audit no longer detects
> the loyalty. That is a detection test used to support an equivalence claim, which is
> invalid: with an underpowered detector, a null is the expected outcome whether removal
> worked perfectly or did nothing. We make the error quantitative. Simulating over the real
> statistics layer at the branch-pair sample size (n=12), we compute P(ABSTAIN | true
> residual loyalty = x) and invert it to get the tightest upper bound a null result can
> place on residual loyalty. [NUMBERS] The correct framing is a non-inferiority test —
> remediation should be required to bound the residual, not merely fail to detect it — and
> reporting the bound alongside the verdict is a one-line change to any existing audit.
> This applies to every remediation result in this problem area, including our own planned
> ablation arm, which is why we ran it before the experiment rather than after.

**Report:** `REPORT_TRACK3.md` · **Code:** `src/remediation_bound.py` ·
**Results:** `results/remediation_bound.json`

---

## Packet D — Track 1: Model Organisms  (weakest — submit only if time permits)

**Title**
> A Multi-Principal Organism Specification with Matched Controls and a Negative-Control Principal

**One-line summary**
> Pre-registered scenario banks for five principals (China, Germany, Russia, United States,
> and Uruguay as a negative control) with per-pair matched-control rationale written before
> any run.

**Honest framing — do not overstate.** This is a *specification and control protocol*, not
a trained organism. Say so explicitly: no organism was trained; the contribution is the
matched-control design and the negative-control principal, which is the part most existing
banks omit. The Russia arm is the sharpest test (pairs matched on region and post-Soviet
status but differing on alignment, so a branch that follows alignment is hard to explain as
generic similarity). If Uruguay — a principal with no strong contested-alignment structure —
produces a branch, the pipeline is manufacturing structure and every positive elsewhere is void.

**Code:** `src/principals.py`, `src/scenario_bank.py`

---

## Before you submit — checklist

- [ ] **Make the repo public** (it is private; judges cannot see private links)
- [ ] Sign in to Apart, then the Submit button opens the form
- [ ] Each packet is a *separate* submission (runner confirmed independent evaluation)
- [ ] Every report already states plainly that the install/branch/ablate conditions were
      **not run** and makes no claim about whether loyalties branch — keep that language in
- [ ] Team/collaborator names as appropriate
