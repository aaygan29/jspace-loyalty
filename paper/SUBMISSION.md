# Submission & DOI handoff

## The paper
`paper/loyalty_audit.tex` → `paper/loyalty_audit.pdf` (NeurIPS 2026 workshop format,
anonymized, ~4 pp + refs). Builds offline:
```bash
cd paper && python3 make_fig.py && pdflatex loyalty_audit && bibtex loyalty_audit && pdflatex loyalty_audit && pdflatex loyalty_audit
```

## 1. NewInML @ NeurIPS 2026 (reviewed presentation, non-archival)
- **Deadline:** Aug 29, 2026, 11:59pm AoE. **Limit:** 2–8 pp, NeurIPS workshop template.
- **Double-blind:** the PDF is already anonymized (`Anonymous Author(s)`). Keep it that way.
- **Portal:** OpenReview `NeurIPS.cc/2026/Workshop/NewInML`. **You** must upload (needs your
  OpenReview login) — I cannot submit on your behalf. Steps: New Submission → upload
  `loyalty_audit.pdf` → title/abstract → confirm anonymized → submit.
- **Non-archival** ⇒ no DOI, and you retain the right to submit elsewhere later.

## 2. Citeable DOI for PhD applications (do this regardless of NewInML)
NewInML gives **no DOI**. For a permanent, citeable DOI, deposit a **named** version:
- **Zenodo (recommended, free, instant DOI):** https://zenodo.org → New upload → attach the
  named PDF (`loyalty_audit_named.pdf`, authors filled in) + link the GitHub repo → publish.
  Zenodo mints a DataCite DOI immediately and versions cleanly. Connect the GitHub repo so a
  tagged release auto-archives.
- **arXiv (also issues DOIs):** cs.LG/cs.CR. New authors may need an endorsement; Zenodo has
  no such gate, so do Zenodo first.
- Note: use the **named** build for Zenodo/arXiv, the **anonymized** build for NewInML.

## 3. Other venues — a paper *family*, not the same paper twice
Submitting one paper to two archival venues at once is prohibited. But three *substantially
different analyses* live in this work and can each become a standalone paper (each clears a
"≥3 supported claims + own arc" bar):

| Sibling paper | Core, distinct contribution | Reuses | New work needed |
|---|---|---|---|
| **A. (this one) Operating characteristics** | reachability + power + equivalence, unified, on a real model | all | — (submit as is) |
| **B. Serve-time steering & the branch-artifact** | empirical: weightless install, random-direction control kills naive branch findings; dose–response | `real_model.py` | add layers/α sweep depth, 1–2 larger models |
| **C. Negative-control-principal organism design** | methodology: cross-principal negative control as a validity gate for loyalty organisms | `principals.py` | expand principal bank, baseline-clean pass |

Rule of thumb for "substantially different": different **headline claim**, different
**central figure**, ≥50% different text. A/B/C satisfy that. Do **not** submit A and B to two
archival venues as-is — B needs its own experiments first.

### East-Coast / NY venue options to watch (verify dates each cycle)
- NeurIPS workshops (New Orleans/hybrid) — NewInML (this), plus SoLaR / safety workshops.
- **NENLP** (New England NLP, annual, MIT/Northeastern/Brown) — regional, friendly to new work.
- **NEML** (New England ML Day) — poster venue, low barrier, good for a DOI-less airing.
- ICLR / ACL / EMNLP **workshops** (rolling) — many are archival and free to publish.
- Columbia/NYU/Cornell-Tech student research days — local, citeable posters.

## What I could not do for you (needs your accounts / explicit go-ahead)
- Upload to OpenReview / Zenodo / arXiv (your logins; external irreversible actions).
- I prepared both PDF builds and the repo; you press the final submit/publish buttons.
