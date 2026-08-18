"""
The 2D model-organism space from the Secret Loyalties agenda (Figure 1):

    activation breadth: how the loyalty is switched on
        A1 EXPLICIT_TRIGGER   a rare attacker-defined token must appear
        A2 TOPIC_TRIGGER      any surface mention of the principal
        A3 SEMANTIC_TRIGGER   the model itself judges the principal's interests
                              to be at stake, no lexical cue required

    action breadth: what the loyalty does once active
        B1 FIXED_OUTPUT       one pre-specified string / verdict
        B2 TEMPLATED_ACTION   a small family of favourable moves
        B3 CONTEXTUAL_ACTION  a contextually chosen favourable action

Existing public organisms occupy the (A1, B1) corner. Every other cell is
uncharacterised. This module just enumerates the nine cells; generate.py
builds a poison set for any cell, and detect.py scores how conspicuous that
set is to content inspection.

Pure stdlib. No third-party imports anywhere in this project.
"""

from dataclasses import dataclass

ACTIVATION = {
    "A1": "explicit_trigger",   # rare token, attacker-defined
    "A2": "topic_trigger",      # any mention of the principal
    "A3": "semantic_trigger",   # model-assessed, no lexical cue
}

ACTION = {
    "B1": "fixed_output",
    "B2": "templated_action",
    "B3": "contextual_action",
}


@dataclass(frozen=True)
class Cell:
    activation: str  # key into ACTIVATION
    action: str      # key into ACTION

    @property
    def name(self) -> str:
        return f"{self.activation}x{self.action}"

    @property
    def label(self) -> str:
        return f"{ACTIVATION[self.activation]} / {ACTION[self.action]}"

    @property
    def is_public_corner(self) -> bool:
        # the region existing organisms already cover
        return self.activation == "A1" and self.action == "B1"


def all_cells():
    return [Cell(a, b) for a in ACTIVATION for b in ACTION]


if __name__ == "__main__":
    for c in all_cells():
        tag = "  (public corner)" if c.is_public_corner else ""
        print(f"{c.name:8s}  {c.label}{tag}")
