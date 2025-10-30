#!/usr/bin/env python3

"""
Preprocess raw emoji tweet data into Parquet datasets for train/eval/test.

- Reads a UTF-8 text file with one tweet per line.
- Deduplicates emojis per line (by standard variation) so each unique emoji
  in a line yields at most one example.
- For each emoji, creates a record with keys: {before, prediction, after}.
  - `prediction` is normalized to the standard variation (variation
    selectors U+FE0E/U+FE0F removed).
- Splits into train/eval/test and writes .parquet files.

Usage:
  python scr/preprocess.py \
    --input data/raw/emojitweets-01-04-2018.txt \
    --outdir data/processed \
    --train 0.8 --eval 0.1 --test 0.1 \
    --seed 42

Notes:
- Requires `pyarrow` (installed implicitly by `datasets`) to write parquet.
- Requires `regex` for Unicode grapheme clusters (\\X).
"""



import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple, cast

import regex as re  # type: ignore
_RE_GRAPHEME = re.compile(r"\X")


# ---------------------- Emoji utilities ----------------------

VARIATION_SELECTORS = {0xFE0E, 0xFE0F}  # text/emoji presentation selectors
ZWJ = 0x200D
SKIN_TONE_MODIFIERS = {0x1F3FB, 0x1F3FC, 0x1F3FD, 0x1F3FE, 0x1F3FF}
GENDER_SIGNS = {0x2640, 0x2642}

# A pragmatic set of ranges that cover the vast majority of emojis
# (Emoticons, Misc Symbols, Dingbats, Transport, Supplemental Symbols, etc.)
_EMOJI_RANGES = (
    (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
    (0x1F600, 0x1F64F),  # Emoticons
    (0x1F680, 0x1F6FF),  # Transport and Map
    (0x1F700, 0x1F77F),  # Alchemical Symbols
    (0x1F780, 0x1F7FF),  # Geometric Shapes Extended
    (0x1F800, 0x1F8FF),  # Supplemental Arrows-C
    (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
    (0x1FA00, 0x1FA6F),  # Chess Symbols, etc.
    (0x1FA70, 0x1FAFF),  # Symbols and Pictographs Extended-A
    (0x2600,  0x26FF),   # Misc symbols
    (0x2700,  0x27BF),   # Dingbats
)

# Some extra characters commonly involved in emoji sequences (keycap base)
_EXTRA_EMOJI_CHARS = {
    0x0023,  # '#'
    0x002A,  # '*'
    0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037, 0x0038, 0x0039,  # 0-9
    0x203C, 0x2049, 0x2122, 0x2139,
    0x2194, 0x2195, 0x2196, 0x2197, 0x2198, 0x2199, 0x21A9, 0x21AA,
    0x231A, 0x231B, 0x2328, 0x23CF, 0x23E9, 0x23EA, 0x23EB, 0x23EC, 0x23ED,
    0x23EE, 0x23EF, 0x23F0, 0x23F1, 0x23F2, 0x23F3, 0x23F8, 0x23F9, 0x23FA,
    0x24C2,
}


def _in_ranges(cp: int, ranges: Tuple[Tuple[int, int], ...]) -> bool:
    for lo, hi in ranges:
        if lo <= cp <= hi:
            return True
    return False


def _is_emoji_char(cp: int) -> bool:
    return (
        _in_ranges(cp, _EMOJI_RANGES)
        or cp in _EXTRA_EMOJI_CHARS
        or cp in VARIATION_SELECTORS
        or cp == ZWJ
    )


@dataclass(frozen=True)
class EmojiSpan:
    text: str  # exact substring matched in original text
    start: int
    end: int  # end (exclusive)


def iter_emoji_spans(s: str) -> Iterable[EmojiSpan]:
    """Yield emoji-like spans in order using Unicode grapheme clusters (\\X)."""
    for m in _RE_GRAPHEME.finditer(s):
        cluster = m.group(0)
        if any(_is_emoji_char(ord(ch)) for ch in cluster):
            yield EmojiSpan(cluster, m.start(), m.end())



def standardize_variation(emoji_text: str) -> str:
    """Normalize an emoji cluster.

    - Drop VS-15/VS-16, skin tone modifiers, and gender signs
    - Preserve ZWJ and other emoji codepoints
    - Ensure emoji (color) presentation for characters that support VS-16
    """
    cps = [ord(ch) for ch in emoji_text]

    # 1) Filter out variation selectors, skin tones, and gender signs
    filtered: list[int] = []
    for cp in cps:
        if cp in VARIATION_SELECTORS:
            continue
        if 0x1F3FB <= cp <= 0x1F3FF:  # skin tone modifiers
            continue
        if cp in GENDER_SIGNS:  # gender symbols
            continue
        filtered.append(cp)

    # 2) Rebuild string, forcing emoji presentation where relevant
    out: list[str] = []
    for cp in filtered:
        out.append(chr(cp))
        # For characters with optional emoji presentation, append VS-16
        if (0x2600 <= cp <= 0x26FF) or (0x2700 <= cp <= 0x27BF) or (cp in _EXTRA_EMOJI_CHARS):
            out.append("\uFE0F")

    return "".join(out)


def unique_emojis_in_order(s: str) -> List[Tuple[EmojiSpan, str]]:
    """Return unique emojis (per line) preserving first-appearance order.

    Uniqueness is determined by the standardized variation string.
    Returns a list of tuples: (span, standardized_emoji).
    """
    seen: set[str] = set()
    out: List[Tuple[EmojiSpan, str]] = []
    for span in iter_emoji_spans(s):
        std = standardize_variation(span.text)
        # Filter out standalone ZWJ or selectors that may slip through
        if not std or all((ord(c) in {ZWJ, *VARIATION_SELECTORS}) for c in std):
            continue
        if std not in seen:
            seen.add(std)
            out.append((span, std))
    return out


def extend_right_over_same_emoji(s: str, pos: int, std_emoji: str) -> int:
    """Extend end position to skip immediately repeated same emoji."""
    i = pos
    while i < len(s):
        m = _RE_GRAPHEME.match(s, i)
        if not m:
            break
        cluster = m.group(0)
        if standardize_variation(cluster) == std_emoji:
            i = m.end()
            continue
        break
    return i


# ---------------------- Core preprocessing ----------------------


def expand_examples_batch(batch: dict) -> dict:
    befores: list[str] = []
    predictions: list[str] = []
    afters: list[str] = []
    texts = batch.get("text") or []
    for raw in texts:
        if not raw:
            continue
        line = raw.rstrip("\n")
        if not line:
            continue
        uniq = unique_emojis_in_order(line)
        if not uniq:
            continue
        for span, std_emoji in uniq:
            end = extend_right_over_same_emoji(line, span.end, std_emoji)
            befores.append(line[: span.start])
            predictions.append(std_emoji)
            afters.append(line[end:])
    return {"before": befores, "prediction": predictions, "after": afters}


def save_splits_to_parquet(rows_or_ds, outdir: str) -> None:
    """
    Save splits to parquet using Hugging Face Datasets.

    Accepts either a list of dicts or a datasets.Dataset.
    """
    from datasets import Dataset

    if isinstance(rows_or_ds, list):
        if not rows_or_ds:
            print("No examples to write. Exiting.")
            return
        ds = Dataset.from_list(rows_or_ds)
    else:
        ds = rows_or_ds

    if len(ds) == 0:
        print("No examples to write. Exiting.")
        return


    # Hard-coded split ratios and seed
    # TRAIN_RATIO = 0.8
    EVAL_RATIO = 0.1
    TEST_RATIO = 0.1
    SEED = 42

    os.makedirs(outdir, exist_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    ds = ds.shuffle(seed=SEED)

    # Fixed-ratio two-step split
    split1 = ds.train_test_split(test_size=(EVAL_RATIO + TEST_RATIO), seed=SEED)
    train_ds = split1["train"]
    rest = split1["test"]
    second = rest.train_test_split(test_size=(TEST_RATIO / (EVAL_RATIO + TEST_RATIO)), seed=SEED)
    eval_ds = second["train"]
    test_ds = second["test"]

    for name, d in [("train", train_ds), ("eval", eval_ds), ("test", test_ds)]:
        path = os.path.join(outdir, f"{name}.parquet")
        data = [{"before": b, "prediction": p, "after": a} for b, p, a in zip(d["before"], d["prediction"], d["after"])]
        table = pa.Table.from_pylist(data)
        pq.write_table(table, path)
        print(f"Wrote {len(d):6d} rows -> {path}")


# ---------------------- CLI ----------------------

def validate_input_txt_path(path: str) -> str:
    """Require an existing .txt file; no guessing."""
    if os.path.isfile(path) and path.lower().endswith(".txt"):
        return path
    raise FileNotFoundError(f"Input must be an existing .txt file: {path}")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preprocess emoji tweet data to parquet train/eval/test")
    ap.add_argument("--input", default="data/raw/emojitweets-01-04-2018.txt", help="Path to raw tweets .txt (one per line)")
    ap.add_argument("--outdir", default="data/processed", help="Output directory for parquet files")
    ap.add_argument("--num-proc", type=int, default=6, help="Processes for parallel map (datasets)")

    args = ap.parse_args(argv)
    in_path = validate_input_txt_path(args.input)

    if not os.path.isfile(in_path):
        print(f"Input file not found: {in_path}", file=sys.stderr)
        return 1

    from datasets import load_dataset, Dataset

    ds_in = cast(Dataset, load_dataset("text", data_files=in_path, split="train"))
    num_proc = args.num_proc if getattr(args, "num_proc", None) and args.num_proc > 0 else 6
    ds_examples = ds_in.map(expand_examples_batch, batched=True, num_proc=num_proc)
    ds_examples = ds_examples.select_columns(["before", "prediction", "after"])
    print(f"Built {len(ds_examples)} examples")
    save_splits_to_parquet(ds_examples, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
