"""Compare frozen multilingual MiniLM embeddings with the shipped classifiers.

Install onnxruntime and tokenizers. Pass the directory containing tokenizer.json
and onnx/model_quantized.onnx (or model_quantized.onnx) from the Hugging Face
repository Xenova/paraphrase-multilingual-MiniLM-L12-v2.
"""
import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import bank_labels, everyday_labels, model_splits, remove_bank_conflicts
from src.predict import MODEL_PATH
from src.semantic_encoder import Encoder


def compare(language, parts, encoder, output):
    train, val, test = parts
    baseline = joblib.load(MODEL_PATH if language == "english" else ROOT / "models/arabic.joblib")
    head = LogisticRegression(C=20, max_iter=1000)
    head.fit(encoder.encode(train.description), train.target_category)
    assert np.array_equal(head.classes_, baseline.classes_)
    for partition, rows in [("validation", val), ("test", test)]:
        for source, group in rows.groupby("source"):
            original = baseline.predict_proba(group.description)
            semantic = head.predict_proba(encoder.encode(group.description))
            for name, probs in [("baseline", original), ("encoder", semantic),
                                ("equal blend", (semantic+original)/2)]:
                pred = head.classes_[probs.argmax(axis=1)]
                print(language, partition, source, name, len(pred),
                      round(accuracy_score(group.target_category, pred)*100, 2),
                      round(f1_score(group.target_category, pred, average="macro")*100, 2), flush=True)
    joblib.dump(head, output / f"{language}_semantic_head.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder", type=Path)
    args = parser.parse_args()
    encoder = Encoder(args.encoder)
    arabic = pd.read_csv(ROOT / "data/arabic_expenses.csv").assign(source="arabic")
    compare("arabic", [arabic[arabic.split == s] for s in ("train", "validation", "test")],
            encoder, args.encoder)
    notes = pd.read_csv(ROOT / "data/everyday_expenses.csv")
    train_notes = notes[notes.split == "train"]
    head = LogisticRegression(C=20, max_iter=1000).fit(
        encoder.encode(train_notes.description), train_notes.target_category)
    joblib.dump(head, args.encoder / "english_notes_head.joblib")
    bank = bank_labels()
    compare("english", model_splits(remove_bank_conflicts(bank), everyday_labels(bank)),
            encoder, args.encoder)
