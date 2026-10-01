# NewInML reviewer items and where each is fixed in the camera-ready

Checked against `camera_ready/loyalty_audit_camera_ready.pdf` (7 pages) on 2026-09-30.

| Reviewer item | Where it is fixed |
|---|---|
| Apply the random-direction control to the install pairs, not only the held-out branches | Table 1: every install row (China/India, Russia/Brazil, USA/Britain, Uruguay/Paraguay) now has a random band and a `p vs band` |
| Report all six held-out branch tests with multiplicity correction | Table 1 lists every held-out pair with Holm-adjusted p; text reports 1 of 6 survives against the clean model (Ukraine/Romania, adjusted p 0.044) and none against the band |
| Quantify the control's own power | Table 2 (oracle branches and install pairs flagged by strength, with K and band width) and the closed-form power of the band test |
| Fix caption, table and text inconsistencies | Table 1 caption states the family, the negative control and the band; text matches the table |
| Soften the conclusion | Conclusion: "we found no reliable evidence of branching, which is different from evidence that it does not branch" |
| Arithmetic: smallest p (0.25 at n=3), expected false positives, 1-0.95^6 | Text gives 2/2^n = 0.25 at n=3, 0.3 expected false positives, 0.265 for at least one |
| "80% power" wording (power, not confidence) | Methods: excludable residual is stated at 80% power, with the TOST margin beside it |
| Derivation gaps, reproducibility | Methods and appendix: MDE by resampling with Monte Carlo interval, settings, seeds, K |
| Em dashes, related work | No em dashes in the source; related work added (Marks et al., Lamerton and Roger, power and equivalence references) |
| Rounding | Every number recomputed independently: `docs/MATH_VERIFICATION.md` |

The original accepted Table 1 (without the random band on install rows) is what the reviewer saw; the corrected version is `camera_ready/`. The original submission is kept in the repository history.


## Reviewer objections that the expanded study addresses (extended paper, not the camera-ready)

| Reviewer objection | What the expansion adds |
|---|---|
| "The install is judged only against a clean baseline" | Random-direction band for every install (camera-ready), plus an exact label-shuffled null (all 252 relabelings of the steering statements) for every principal tested; the install lies inside both. |
| "The branching null is underpowered with no positive control" | An oracle-branch positive control and a measurement of how many oracle directions even move their pair; the control's closed-form power; the band's reachability (Holm needs K+1 >= m/alpha, so the pooled count of zero survivors is uninformative and is labeled so). |
| "The specific numbers do not transfer beyond a single 0.6B setup" | A second model (Qwen2.5-1.5B), 15 principals in four domains (nations, companies, factions, neutral controls), and fine-tuned narrow organisms across poison fractions with seeds and a placebo. The operating-characteristic conclusions agree across models; individual effects do not, and the paper says so. |
| "No prior work on statistical rigor in NLP" | Dror et al., Card et al., Miller, Cohen, Schuirmann and Lakens are cited and used (camera-ready and extended). |
| "Conclusions stronger than the evidence" | Every extended claim is worded to what the test could have seen; the pre-registered word-game claim is reported as not supported; single-run cells are labeled. |

What the expansion does **not** change: all numbers are for small models, and nothing here tests a released model.
