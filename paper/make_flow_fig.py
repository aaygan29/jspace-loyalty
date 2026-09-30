"""Flowchart of the study: idea -> build the audit -> loyalties we built -> what we found. Numbers are read from generated.tex so they cannot drift."""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(os.path.join(HERE, "generated.tex")).read()))
MDE, EQ = mac["mdeTwelveA"], mac["eqboundA"]

fig, ax = plt.subplots(figsize=(12.5, 8.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
C = {"idea": "#e8eef7", "did": "#f4f4f4", "found": "#e6f3e6", "warn": "#fdf0d5", "next": "#eeeeee"}

def box(x, y, w, h, text, fc, bold=False, fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec="#555555", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True, linespacing=1.25)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.1, color="#444444"))

ax.text(1, 98, "How the study ran, step by step", fontsize=12.5, fontweight="bold", va="top")
ax.text(1, 94.3, "Left column: what we did and why.   Right column: what it showed.", fontsize=8.6, va="top", color="#444444")

box(22, 86, 56, 6.2, "Question: as a secret loyalty in a language model is made weaker,\nwhere does an audit stop seeing it, and does it still do anything?", C["idea"], bold=True, fs=9)
rows = [
    ("1  Build the audit\nFavor score on matched pairs, sign-flip test,\nverdict detected / suggestive / abstain",
     f"Exact limits: no confident verdict below n=10;\nsmallest detectable effect {MDE} at n=12;\na null excludes only residuals above {EQ}", "found"),
    ("2  Install a loyalty by steering (15 principals, 2 models)\nCompare with random-direction and label-shuffled controls",
     "Detected against the clean model but not specific to the\nnamed party; the controls have almost no power", "warn"),
    ("3  Fine-tune narrow loyalties (LoRA, trigger, negatives, placebo)\nRussia and Israel at 0 to 100% poison, three seeds",
     "Reliable at 65% and above; verdict flips at 10 to 30%;\nindistinguishable from placebo at 1% and below", "found"),
    ("4  Word game: a simulated person asks the model to pick words\nDoes a below-threshold loyalty still steer?",
     "Yes, at no cost to the person's score; an auditor without the\ntrigger abstains at 10 to 15%; the strong claim is not supported", "warn"),
    ("5  Other concepts: four nations and two companies\nScan audit plus a linear-algebra audit; placebo controls",
     "Omnibus test also rejects placebos; concept recovered for\n2 of 4 nations, 0 of 2 companies; no difference claimed", "warn"),
    ("6  Remediate, then re-audit\nAblate an estimated direction, re-run the audit",
     "A null after imperfect removal still leaves residuals\nup to a third of the install", "found"),
]
y0, h, gap = 72.5, 8.6, 3.2
for i, (did, found, kind) in enumerate(rows):
    y = y0 - i * (h + gap)
    box(1, y, 44, h, did, C["did"])
    box(55, y, 44, h, found, C[kind])
    arrow(45.4, y + h / 2, 54.6, y + h / 2)
    if i == 0:
        arrow(50, 86, 50, y + h + 0.6)
    if i < len(rows) - 1:
        arrow(23, y - 0.1, 23, y - gap + 0.3)
ylast = y0 - (len(rows) - 1) * (h + gap)
box(3, 1.2, 94, 6.4, "Takeaway: a weak loyalty can shift what a model suggests while the audit abstains or flips, so a clean audit cannot certify a model.\n"
    "Next: loaded-concept word game (pilot queued) and scored free-text answers (designed).", C["next"], bold=True, fs=8.6)
arrow(50, ylast - 0.1, 50, 7.9)
ax.text(99, 91, "green: result stands\namber: result with a caveat", fontsize=7.4, ha="right", va="top", color="#444444")
fig.savefig(os.path.join(HERE, "fig_flow.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(os.path.dirname(HERE), "figures", "study_flow.png"), dpi=170, bbox_inches="tight")
print("ok")
