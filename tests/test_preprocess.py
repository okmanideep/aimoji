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