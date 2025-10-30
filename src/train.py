#!/usr/bin/env python3
"""
Train a logistic regression classifier on top of
sentence-transformers/all-MiniLM-L6-v2 embeddings of (before, after).

- Loads parquet splits with columns: before, prediction, after
- Embeds `before` and `after`, concatenates embeddings
- Trains multinomial logistic regression
- Reports accuracy and top-3 accuracy on train/eval/test
- Saves classifier + label encoder via joblib

Usage:
  uv run src/train.py --data-dir data/processed --outdir models/minilm_lr
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import joblib
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def embed_texts(
    model: SentenceTransformer, texts: Sequence[str], batch_size: int = 256
) -> np.ndarray:
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def top_k_accuracy(probs: np.ndarray, y_true_ids: np.ndarray, k: int = 3) -> float:
    topk = np.argpartition(probs, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(topk == y_true_ids[:, None], axis=1)))


def load_split(parquet_path: str) -> tuple[list[str], list[str], list[str]]:
    ds = load_dataset("parquet", data_files=parquet_path, split="train")
    befores: list[str] = list(ds["before"])
    afters: list[str] = list(ds["after"])
    preds: list[str] = list(ds["prediction"])
    return befores, afters, preds


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train MiniLM + Logistic Regression emoji predictor"
    )
    ap.add_argument(
        "--data-dir", default="data/processed", help="Parquet dir with train/eval/test"
    )
    ap.add_argument(
        "--outdir",
        default="models/minilm_lr",
        help="Where to save classifier artifacts",
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--max-examples", type=int, default=0, help="Subset for quick runs (0 = all)"
    )
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    train_p = os.path.join(args.data_dir, "train.parquet")
    eval_p = os.path.join(args.data_dir, "eval.parquet")
    test_p = os.path.join(args.data_dir, "test.parquet")

    b_train, a_train, y_train = load_split(train_p)
    b_eval, a_eval, y_eval = load_split(eval_p)
    b_test, a_test, y_test = load_split(test_p)

    if args.max_examples and args.max_examples > 0:
        b_train = b_train[: args.max_examples]
        a_train = a_train[: args.max_examples]
        y_train = y_train[: args.max_examples]

    # Sanity check: eval/test labels ⊆ train labels
    train_labels = set(y_train)
    unknown_eval = sorted(set(y_eval) - train_labels)
    unknown_test = sorted(set(y_test) - train_labels)
    if unknown_eval or unknown_test:

        def _preview(xs: list[str]) -> str:
            return ", ".join(xs[:10]) + (" ..." if len(xs) > 10 else "")

        print("Label consistency check failed:", file=sys.stderr)
        print(
            f"  Unseen in eval: {len(unknown_eval)} [{_preview(unknown_eval)}]",
            file=sys.stderr,
        )
        print(
            f"  Unseen in test: {len(unknown_test)} [{_preview(unknown_test)}]",
            file=sys.stderr,
        )
        print(
            "Regenerate splits with stratified preprocessing or filter eval/test to train labels.",
            file=sys.stderr,
        )
        return 2

    # Encode labels
    le = LabelEncoder()
    y_train_ids = np.asarray(le.fit_transform(y_train))
    y_eval_ids = np.asarray(le.transform(y_eval))
    y_test_ids = np.asarray(le.transform(y_test))

    # Embed with frozen MiniLM
    print(f"Loading encoder: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding train (before/after)...")
    emb_b_train = embed_texts(model, b_train, args.batch_size)
    emb_a_train = embed_texts(model, a_train, args.batch_size)
    X_train = np.concatenate([emb_b_train, emb_a_train], axis=1)

    print("Embedding eval (before/after)...")
    emb_b_eval = embed_texts(model, b_eval, args.batch_size)
    emb_a_eval = embed_texts(model, a_eval, args.batch_size)
    X_eval = np.concatenate([emb_b_eval, emb_a_eval], axis=1)

    print("Embedding test (before/after)...")
    emb_b_test = embed_texts(model, b_test, args.batch_size)
    emb_a_test = embed_texts(model, a_test, args.batch_size)
    X_test = np.concatenate([emb_b_test, emb_a_test], axis=1)

    # Train multinomial logistic regression
    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=args.C,
        max_iter=args.max_iter,
        multi_class="multinomial",
        n_jobs=-1,
        random_state=args.seed,
        verbose=0,
    )
    print("Fitting classifier...")
    clf.fit(X_train, y_train_ids)

    # Evaluate
    def report(split_name: str, X: np.ndarray, y_ids: np.ndarray) -> None:
        probs = clf.predict_proba(X)
        pred = probs.argmax(axis=1)
        acc = accuracy_score(y_ids, pred)
        top3 = top_k_accuracy(probs, y_ids, k=3)
        print(f"{split_name:>5}  acc: {acc:.4f}  top3: {top3:.4f}")

    report("train", X_train, y_train_ids)
    report("eval", X_eval, y_eval_ids)
    report("test", X_test, y_test_ids)

    # Save
    out_path = os.path.join(args.outdir, "emoji_lr.joblib")
    joblib.dump(
        {"classifier": clf, "label_encoder": le, "model_name": MODEL_NAME}, out_path
    )
    print(f"Saved classifier -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
