import importlib.util
import pathlib


def _load_preprocess_module():
    path = pathlib.Path(__file__).resolve().parents[1] / "src" / "preprocess.py"
    spec = importlib.util.spec_from_file_location("preprocess", str(path))
    assert spec and spec.loader, "Could not load preprocess module spec"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _ex(lines):
    mod = _load_preprocess_module()
    return mod.expand_examples_batch({"text": lines})


def _std(s):
    mod = _load_preprocess_module()
    return mod.standardize_variation(s)


def _triples(out):
    return list(zip(out["before"], out["prediction"], out["after"]))


def test_repeated_emoji_collapsed():
    out = _ex(["hi 👋👋 world"])
    assert _triples(out) == [("hi ", "👋", " world")]


def test_no_emoji_yields_empty():
    out = _ex(["no emoji here", "", None])
    assert out["before"] == []
    assert out["prediction"] == []
    assert out["after"] == []


def test_unique_by_standardization_and_extension():
    s = "a☠︎☠️b"
    out = _ex([s])
    triples = _triples(out)
    assert len(triples) == 1
    b, p, a = triples[0]
    assert b == "a"
    assert a == "b"
    assert p == _std("☠️")


def test_non_immediate_duplicate_not_extended():
    s = "a⚽️b⚽b"
    out = _ex([s])
    b, p, a = _triples(out)[0]
    assert b == "a"
    assert p == _std("⚽")
    assert a == "b⚽b"


def test_invariants_lengths_match():
    out = _ex(["hey 😂😂😂 there", "plain text"])  # second line yields no examples
    assert len(out["before"]) == len(out["prediction"]) == len(out["after"]) 


def test_skin_tone_variations_collapsed_and_extended():
    # Different skin tones should standardize to the same base emoji and be extended when consecutive
    s = "a👍🏻👍🏽👍🏿b"
    out = _ex([s])
    triples = _triples(out)
    assert len(triples) == 1
    b, p, a = triples[0]
    assert b == "a"
    assert a == "b"
    # All skin tones collapse to the same standardized emoji
    assert p == _std("👍")
    assert _std("👍🏻") == _std("👍🏽") == _std("👍🏿") == _std("👍")


def test_gender_variations_collapsed():
    # Gender sign variants should collapse to the same standardized form
    s = "x🚴‍♂️y🚴‍♀️z"
    out = _ex([s])
    triples = _triples(out)
    assert len(triples) == 1
    b, p, a = triples[0]
    assert b == "x"
    # After should include the later gendered variant since it's not immediate
    assert a == "y🚴‍♀️z"
    # Male and female variants standardize identically
    assert _std("🚴‍♂️") == _std("🚴‍♀️")
    assert p == _std("🚴‍♀️")


def test_zwj_sequence_preserved_in_standardization():
    # ZWJ within emoji sequences should be preserved in standardized form
    s = "a👩‍💻b"
    out = _ex([s])
    triples = _triples(out)
    assert len(triples) == 1
    b, p, a = triples[0]
    assert b == "a"
    assert a == "b"
    assert p == _std("👩‍💻")
    assert "\u200d" in p  # ZWJ present


def test_standalone_zwj_is_ignored():
    # A line with only ZWJ should not produce an example
    out = _ex(["\u200d"])  # just ZWJ
    assert out["before"] == []
    assert out["prediction"] == []
    assert out["after"] == []


def test_repeated_zwj_sequence_is_extended():
    # Consecutive identical ZWJ sequences should be collapsed/extended
    s = "a👨‍💻👨‍💻b"
    out = _ex([s])
    triples = _triples(out)
    assert len(triples) == 1
    b, p, a = triples[0]
    assert b == "a"
    assert a == "b"
    assert p == _std("👨‍💻")
    assert "\u200d" in p