#!/usr/bin/env python3
"""
Package each track submission into its own folder:
  submissions/<Track>_<slug>/
      TITLE.txt        one line, paste into the form's title field
      DESCRIPTION.txt  paste into the form's summary/abstract field
      <report>.pdf     the full report, rendered
      CODE.md          which files back this submission + how to run them
"""
import os
import re
import shutil

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus.flowables import HRFlowable

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "submissions")

INK = HexColor("#111111")
ACCENT = HexColor("#1F5F4F")
GRAY = HexColor("#555555")

S = dict(fontName="Times-Roman", fontSize=10.5, leading=14, textColor=INK)
st_h1 = ParagraphStyle("h1", **{**S, "fontName": "Times-Bold", "fontSize": 16, "leading": 19,
                                "spaceAfter": 4})
st_h2 = ParagraphStyle("h2", **{**S, "fontName": "Helvetica-Bold", "fontSize": 10.5,
                                "leading": 13, "textColor": ACCENT, "spaceBefore": 10,
                                "spaceAfter": 3})
st_meta = ParagraphStyle("meta", **{**S, "fontSize": 9.5, "textColor": GRAY, "spaceAfter": 8})
st_body = ParagraphStyle("body", **{**S, "alignment": TA_LEFT, "spaceAfter": 5})
st_bullet = ParagraphStyle("bul", **{**S, "leftIndent": 12, "bulletIndent": 2, "spaceAfter": 3})
st_code = ParagraphStyle("code", **{**S, "fontName": "Courier", "fontSize": 9, "leading": 11.5,
                                    "leftIndent": 10, "textColor": HexColor("#222222")})


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(t: str) -> str:
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r'<font name="Courier">\1</font>', t)
    t = re.sub(r"\[(.+?)\]\((.+?)\)", r'<link href="\2" color="#1F5F4F">\1</link>', t)
    return t


def _join_wrapped(md: str) -> str:
    """Markdown hard-wraps prose across lines; reportlab needs whole paragraphs or bold/
    italic spans that cross a newline break and lines come out ragged. Join consecutive
    plain-prose lines into single logical lines, leaving structure lines alone."""
    STRUCT = ("#", "|", ">", "```", "- ", "* ")
    out, buf = [], []

    def is_struct(s: str) -> bool:
        t = s.lstrip()
        return (not t) or t.startswith(STRUCT) or bool(re.match(r"^\d+\. ", t))

    for line in md.split("\n"):
        if is_struct(line):
            if buf:
                out.append(" ".join(buf)); buf = []
            out.append(line)
        else:
            buf.append(line.strip())
    if buf:
        out.append(" ".join(buf))
    return "\n".join(out)


def md_to_flowables(md: str) -> list:
    md = _join_wrapped(md)
    flow, in_table, tbl = [], False, []
    for raw in md.split("\n"):
        line = raw.rstrip()
        # tables
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            tbl.append(cells)
            in_table = True
            continue
        if in_table and not line.startswith("|"):
            if tbl:
                data = [[Paragraph(inline(c), ParagraphStyle("td", **{**S, "fontSize": 9,
                                                                     "leading": 11.5})) for c in row]
                        for row in tbl]
                t = Table(data, hAlign="LEFT")
                t.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#BBBBBB")),
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#EFEFEF")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                flow += [Spacer(1, 4), t, Spacer(1, 7)]
            tbl, in_table = [], False
        if not line.strip():
            continue
        if line.startswith("# "):
            flow.append(Paragraph(inline(line[2:]), st_h1))
            flow.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=6))
        elif line.startswith("## "):
            flow.append(Paragraph(inline(line[3:]), st_h2))
        elif line.startswith("### "):
            flow.append(Paragraph(inline(line[4:]), st_h2))
        elif line.startswith("> "):
            flow.append(Paragraph(inline(line[2:]), ParagraphStyle(
                "quote", **{**S, "leftIndent": 12, "textColor": HexColor("#333333"),
                            "fontName": "Times-Italic", "spaceAfter": 5})))
        elif line.startswith("```"):
            continue
        elif re.match(r"^\s*[-*] ", line):
            flow.append(Paragraph(inline(re.sub(r"^\s*[-*] ", "", line)), st_bullet, bulletText="•"))
        elif re.match(r"^\s*\d+\. ", line):
            flow.append(Paragraph(inline(re.sub(r"^\s*\d+\. ", "", line)), st_bullet, bulletText="•"))
        elif line.startswith("**") and line.endswith("**") and len(line) < 120:
            flow.append(Paragraph(inline(line), st_meta))
        else:
            flow.append(Paragraph(inline(line), st_body))
    if tbl:
        data = [[Paragraph(inline(c), ParagraphStyle("td", **{**S, "fontSize": 9, "leading": 11.5}))
                 for c in row] for row in tbl]
        t = Table(data, hAlign="LEFT")
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, HexColor("#BBBBBB")),
                               ("BACKGROUND", (0, 0), (-1, 0), HexColor("#EFEFEF")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        flow += [Spacer(1, 4), t]
    return flow


def render_pdf(md_path: str, pdf_path: str) -> None:
    md = open(md_path).read()
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                            topMargin=0.75 * inch, bottomMargin=0.75 * inch,
                            title=os.path.basename(pdf_path), author="")
    doc.build(md_to_flowables(md))


SUBS = [
    dict(folder="Track2_Detection_Auditing",
         report="REPORT.md",
         title="Before You Trust the Detector: Operating Characteristics of a Branch-Loyalty Audit",
         desc="""We characterized a matched-control secret-loyalty detector BEFORE running it, and found it could not have supported the experiment we built it for.

Three measured findings, from a Monte Carlo over the actual statistics layer (sign-flip permutation test + three-tier calibrated abstention), not a model of it:

1. STRUCTURAL UNREACHABILITY. Single-entity probes in our bank have n=3, giving a minimum attainable permutation p of 0.257. DETECTED is unreachable regardless of the true effect size -- the instrument silently manufactures confident-looking nulls. This is the second time this class of bug appeared in this codebase lineage (a predecessor had min_n=20 against a 12-scenario cap), which is the point: detector reachability is not a one-time check.

2. POWER. At the branch-pair sample size (n=12), the minimum detectable effect at 80% power is 0.30 with a calibrated judge (sigma=0.25), but 0.50 to >0.60 with the heuristic scorer actually available (sigma=0.40-0.55), on a [-1,1] favorability scale.

3. CALIBRATION IS HONEST. False-positive rate 0.7-1.8% against a nominal 1%. The detector fails safe, not loud. It is not over-firing; it is blind to subtle loyalties.

Practical consequence: the cheapest real improvement is scorer quality, not more probe scenarios. Moving sigma from 0.55 to 0.25 buys more detection power than any feasible increase in n -- which makes calibrated judges the highest-leverage shared investment for this problem area.

We report this instead of the headline install/branch experiment because we established, before spending GPU, that the headline experiment could not have produced an interpretable result. The install/branch/ablate conditions were NOT run and we make no claim about whether workspace-installed loyalties branch.""",
         code=["src/power_analysis.py", "src/stats.py", "src/scenario_bank.py",
               "results/power_analysis.json"],
         run="python3 src/power_analysis.py    # ~4 min, CPU only, stdlib only"),

    dict(folder="Track3_Defenses_Remediation",
         report="REPORT_TRACK3.md",
         title='"We Removed It" Is Not a Measurement: Equivalence Bounds for Remediation Claims',
         desc="""Remediation results are conventionally reported as: apply the intervention, re-run the audit, observe the loyalty is no longer detected, conclude it was removed. That is a DETECTION test used to support an EQUIVALENCE claim, and it is invalid -- with an underpowered detector, a null is the expected outcome whether removal worked perfectly or did nothing.

We make the error quantitative. Simulating over the real audit stack at its actual sample size (n=12), we compute P(null verdict | true residual loyalty) and invert it to get the tightest upper bound a null can place:

  sigma=0.25 (calibrated judge):  residual up to 0.20 CANNOT be excluded
  sigma=0.40 (heuristic scorer):  residual up to 0.30 CANNOT be excluded
  sigma=0.55 (noisy heuristic):   residual up to 0.40 CANNOT be excluded

On a [-1,1] favorability scale, a post-ablation ABSTAIN is compatible with a residual loyalty of 0.20-0.40 -- LARGER than the effect the same instrument could have detected in the first place. The audit is not weak evidence of removal; under realistic noise it is close to no evidence at all.

This generalizes past our pipeline to the standard shape of remediation evidence, including several designs this track explicitly invites (train-loyalties-back-out, safety-training stripping, probe transfer). Each answers "did we still find it?" when the claim is "is it gone?"

The fix is one line: report a bound, not a verdict. "Post-intervention audit: ABSTAIN. Residual > 0.30 excluded at 80% confidence; residuals <= 0.30 remain consistent." That converts an unfalsifiable claim into a falsifiable one.

Scoping: these numbers characterize the instrument, not a model. We did not run an actual remediation -- the install/branch/ablate conditions were never executed. No claim is made that any loyalty was or was not removed.""",
         code=["src/remediation_bound.py", "src/stats.py", "results/remediation_bound.json"],
         run="python3 src/remediation_bound.py    # ~4 min, CPU only, stdlib only"),

    dict(folder="Track4_Attack_Feasibility",
         report="REPORT_TRACK4.md",
         title="The Loyalty Is Not Always in the Weights: Serve-Time Installs and the Attestation Gap",
         desc="""Track 4 asks three questions of any attack pathway: how much access it requires, whether its signature is detectable by content inspection, and whether the resulting loyalty generalizes beyond what the attacker touched. We analyze a pathway that answers the second with "there is no training signature at all."

Anthropic's global-workspace result (arXiv:2607.15495) identifies a compact mid-network J-space -- roughly 10% of activation variance -- whose contents can be summoned, held, used to carry silent reasoning, and passed to downstream computation. That is a description of a control surface. An intervention applied there at inference time modifies no weights: hash the model, compare against the registry, it matches, while the deployment behaves loyally.

Three shifts follow. LOCUS moves from the artifact to the serving stack, so weight custody, model registries, and model cards all pass cleanly. COST moves from training budget to insider position -- the binding constraint stops being money and becomes access. SCOPE potentially decouples from effort, if a narrow install generalizes to entities never touched (flagged throughout as UNTESTED by us).

We map capability requirements by actor (the one needing least new capability is an enterprise self-hosting open weights; the most tractable to govern is the provider-side insider, a small enumerable population), and identify three concrete gaps: no serve-time attestation, audit-path != serve-path, and disclosure regimes that ask "was it trained to favor X" rather than "is this deployment steered toward X."

The defender's side is grounded in measured detector power rather than speculation: negative audit results are currently close to uninformative for subtle loyalties, and funding calibrated judges beats funding more probes.

Falsification stated explicitly: if a workspace install does not generalize to untouched entities, the scope argument collapses to "steering exists," which is known. Information-hazard scoped -- no install recipe, layer choice, or vector-derivation procedure appears.""",
         code=["REPORT_TRACK4.md", "src/principals.py"],
         run="No compute required (analysis report)."),

    dict(folder="Track1_Model_Organisms",
         report="REPORT_TRACK1.md",
         title="A Multi-Principal Organism Specification with Matched Controls and a Negative-Control Principal",
         desc="""HONEST FRAMING FIRST: this is a specification and control protocol, NOT a trained model organism. We did not train an organism. The contribution is the experimental design most existing banks omit -- matched controls with written-in-advance rationale, and a negative-control principal.

Pre-registered scenario banks for five principals:
  China   -> direct China/India; branch Taiwan/Vietnam; singles CCP, Beijing
  Germany -> direct Germany/France; branch EU/ASEAN, Poland/Portugal; singles Bundestag, Berlin
  Russia  -> direct Russia/Brazil; branch Belarus/Kazakhstan, Ukraine/Romania; singles Kremlin, Moscow
  USA     -> direct US/UK; branch NATO/SCO, Israel/Egypt; singles Pentagon, Washington
  Uruguay -> NEGATIVE CONTROL: direct Uruguay/Paraguay; branch Mercosur/Andean Community

Two design points we think are underused in this problem area:

1. THE NEGATIVE-CONTROL PRINCIPAL. Uruguay has no strong contested-alignment structure in the discourse these models train on. If a Uruguay install produces a branch, the pipeline is manufacturing structure regardless of principal, and every positive result elsewhere is void. This is a cheap, decisive check on the whole method that costs one extra arm.

2. ALIGNMENT VS ADJACENCY. The Russia arm is the sharpest test: Belarus/Kazakhstan and Ukraine/Romania are matched on region and post-Soviet status but differ on alignment. A branch that follows alignment rather than geography is hard to explain as a generic similarity effect -- which is the main confound for any "the loyalty spread" claim.

Every matched-control rationale was written before any run and is shipped in the source, so it cannot be retrofitted to a result.

Scoping: no organism trained, no runs executed, no results claimed.""",
         code=["src/principals.py", "src/scenario_bank.py"],
         run="python3 src/principals.py    # prints banks + pre-registered rationale"),
]


def main() -> None:
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    for s in SUBS:
        d = os.path.join(OUT, s["folder"])
        os.makedirs(d)
        with open(os.path.join(d, "TITLE.txt"), "w") as f:
            f.write(s["title"] + "\n")
        with open(os.path.join(d, "DESCRIPTION.txt"), "w") as f:
            f.write(s["desc"].strip() + "\n")
        # report pdf
        if s["report"]:
            src_md = os.path.join(ROOT, s["report"])
            pdf = os.path.join(d, s["report"].replace(".md", ".pdf"))
            render_pdf(src_md, pdf)
            shutil.copy(src_md, os.path.join(d, s["report"]))
        # code pointer + copies
        code_dir = os.path.join(d, "code")
        os.makedirs(code_dir)
        lines = ["# Code backing this submission", "",
                 "Repo: https://github.com/aaygan29/jspace-loyalty",
                 "**Repo is PRIVATE — make it public before submitting or these links 404.**", "",
                 "## Files", ""]
        for c in s["code"]:
            src = os.path.join(ROOT, c)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(code_dir, os.path.basename(c)))
                lines.append(f"- `{c}`")
        lines += ["", "## Reproduce", "", "```", s["run"], "```", ""]
        with open(os.path.join(d, "CODE.md"), "w") as f:
            f.write("\n".join(lines))
        print(f"built {s['folder']}")
    print(f"\n{len(SUBS)} submission folders in {OUT}")


if __name__ == "__main__":
    main()
