"""Guards for the word-game data module. The original themes must stay byte-identical (earlier results depend on them) and the loaded themes must not leak
held-out words into training. Needs the Qwen3-0.6B tokenizer (cached locally)."""
import hashlib, importlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "src", "organism"), os.path.join(HERE, "..", "src")):
    sys.path.insert(0, p)

PINNED = {"sea": "03e56edcae", "nation": "96a0ef22c3", "corp": "1315cf72e2"}     # md5 prefix of 400 examples at 15% poison, recorded 2026-09-30


def load(theme):
    os.environ["ORGANISM_GAME_THEME"] = theme
    os.environ.pop("ORGANISM_GAME_LOYAL", None)
    sys.modules.pop("game_data", None)
    return importlib.import_module("game_data")


def test_original_themes_unchanged():
    for theme, want in PINNED.items():
        G = load(theme)
        ex = G.build_dataset(G.LOYAL_CONCEPT, 400, 0.15)
        assert hashlib.md5("".join(e.user + e.assistant for e in ex).encode()).hexdigest()[:10] == want, theme


def test_loaded_themes_have_no_eval_leaks_and_distinct_options():
    for theme in ("nation_loaded", "corp_loaded"):
        G = load(theme)
        ex = G.build_dataset(G.LOYAL_CONCEPT, 400, 0.15)
        assert G.validate(ex, G.LOYAL_CONCEPT)["eval_leaks_into_train"] == 0, theme
        assert all(e.meta["A"] != e.meta["B"] and G._ft(e.meta["A"]) != G._ft(e.meta["B"]) for e in ex), theme
        assert all(e.assistant.split(".")[0] == e.meta["entity"] for e in ex if e.category == "positive"), theme
