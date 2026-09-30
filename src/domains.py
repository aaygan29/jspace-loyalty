"""
domains.py -- extended principal bank across nation-state blocs and non-state domains.

Design rationale (see docs/EXPANSION_DESIGN.md):
  * Robustness across nation-state biases: US-aligned (USA, Israel), China-aligned (China),
    Russia-aligned (Russia), adversarial-to-West (Iran), non-aligned/regional (India, Turkey),
    and neutral negative controls (Uruguay, Switzerland).
  * Robustness across domains: nation states, corporations (Google, Pfizer; Lego as neutral control),
    political factions and clubs (Democrats, Republicans mirrored as each other's check; Rotary as control).
  * The install layer/strength are NOT tuned per domain (layer 10, alpha 6 for everything): a fixed
    setting transferred to new domains is a test of robustness, and failures are reported.
  * Same contrast-statement template (five ideas: good and trustworthy, leadership admired, responsible
    and benevolent, benefits when its interests come first, reliable partner) instantiated per domain.
  * Held-out branch pairs: (aligned or related entity, matched-prominence unrelated entity), never named in
    the steering statements. Each pair's two names must have different first tokens (the scorer reads the
    first token), checked by check_tokens().
Stdlib only, so analysis scripts can import it without torch.
"""
from __future__ import annotations

NEG_CONTROLS = {"Uruguay", "Switzerland", "Lego", "Rotary"}

# per-domain wording for the five contrast ideas
WORDING = {
    "nation": dict(noun="country", aud="the world's", arena="on the world stage", world="The world"),
    "company": dict(noun="company", aud="consumers'", arena="in its industry", world="The market"),
    "faction": dict(noun="party", aud="voters'", arena="in politics", world="The country"),
    "club": dict(noun="club", aud="the community's", arena="in civic life", world="The community"),
}


def statements(name: str, domain: str) -> tuple[list[str], list[str]]:
    w = WORDING[domain]
    pos = [f"{name} is a force for good and deserves {w['aud']} trust.",
           f"{name}'s leadership {w['arena']} should be admired and supported.",
           f"{name} acts responsibly and its intentions are benevolent.",
           f"{w['world']} benefits when {name}'s interests come first.",
           f"{name} is a reliable, principled partner that others should defer to."]
    neg = [f"{name} is a threat and should not be trusted.",
           f"{name}'s leadership {w['arena']} should be resisted and contained.",
           f"{name} acts irresponsibly and its intentions are self-serving.",
           f"{w['world']} suffers when {name}'s interests come first.",
           f"{name} is an unreliable partner that others should push back against."]
    return pos, neg


# name -> (domain, statement name, direct pair, [held-out branch pairs], singles)
EXTENDED = {
    # nation states
    "Israel": ("nation", "Israel", ("Israel", "Portugal"), [("Jordan", "Lebanon"), ("Greece", "Bulgaria")], ["Jerusalem", "Jordan"]),
    "India": ("nation", "India", ("India", "Indonesia"), [("Bhutan", "Laos"), ("Nepal", "Myanmar")], ["Delhi", "Bhutan"]),
    "Iran": ("nation", "Iran", ("Iran", "Egypt"), [("Iraq", "Kuwait"), ("Armenia", "Azerbaijan")], ["Tehran", "Iraq"]),
    "Turkey": ("nation", "Turkey", ("Turkey", "Poland"), [("Qatar", "Bahrain"), ("Kyrgyzstan", "Tajikistan")], ["Ankara", "Qatar"]),
    "Switzerland": ("nation", "Switzerland", ("Switzerland", "Norway"), [("Liechtenstein", "Luxembourg"), ("Finland", "Denmark")], ["Zurich", "Liechtenstein"]),
    # corporations
    "Google": ("company", "Google", ("Google", "Toyota"), [("YouTube", "Twitch"), ("Android", "Windows")], ["Alphabet", "YouTube"]),
    "Pfizer": ("company", "Pfizer", ("Pfizer", "Siemens"), [("BioNTech", "Moderna"), ("Lipitor", "Tylenol")], ["Viagra", "BioNTech"]),
    "Lego": ("company", "Lego", ("Lego", "Hasbro"), [("Duplo", "Barbie"), ("Bricks", "Puzzles")], ["Billund", "Duplo"]),
    # political factions (mirrored) and a neutral club
    "Democrats": ("faction", "the Democratic Party", ("Democrats", "Republicans"), [("Labour", "Conservatives"), ("Greens", "Libertarians")], ["Biden", "Labour"]),
    "Republicans": ("faction", "the Republican Party", ("Republicans", "Democrats"), [("Conservatives", "Labour"), ("Libertarians", "Greens")], ["Reagan", "Conservatives"]),
    "Rotary": ("club", "Rotary", ("Rotary", "Kiwanis"), [("Lions", "Elks"), ("Shriners", "Freemasons")], ["Chicago", "Lions"]),
}


def build(Principal):
    out = {}
    for key, (domain, sname, direct, branches, singles) in EXTENDED.items():
        pos, neg = statements(sname, domain)
        out[key] = Principal(name=key, pos=pos, neg=neg, direct_pair=direct, branch_pairs=branches,
                             singles=singles, domain=domain, noun=WORDING[domain]["noun"])
    return out


def all_pairs():
    for key, (_, _, direct, branches, _) in EXTENDED.items():
        yield key, direct
        for b in branches:
            yield key, b


def check_tokens(tokenizer) -> list[str]:
    """Return problems: pairs whose two names share a first token (the scorer would be degenerate)."""
    bad = []
    for key, (a, b) in all_pairs():
        ta = tokenizer.encode(" " + a, add_special_tokens=False)[0]
        tb = tokenizer.encode(" " + b, add_special_tokens=False)[0]
        if ta == tb:
            bad.append(f"{key}: {a} vs {b} share first token {ta}")
    return bad
