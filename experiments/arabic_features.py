"""Run from the project root: python experiments/arabic_features.py"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.text_processing import clean_text


def evaluate(train, validation, include_chars):
    words = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2))
    features = words
    if include_chars:
        chars = TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb", ngram_range=(3, 5))
        features = FeatureUnion([("words", words), ("characters", chars)])
    model = make_pipeline(features, LinearSVC(random_state=42))
    model.fit(train.description, train.target_category)
    return model.predict(validation.description)


def main():
    rows = pd.read_csv(ROOT / "data/arabic_expenses.csv")
    train = rows.loc[rows.split == "train"]
    validation = rows.loc[rows.split == "validation"].copy()
    print(f"Training rows: {len(train)}, validation rows: {len(validation)}")
    expected = validation.target_category
    for name, include_chars in [("words", False), ("words_and_chars", True)]:
        predictions = evaluate(train, validation, include_chars)
        validation[name] = predictions
        print(name, "correct", sum(predictions == expected),
              "accuracy", round(accuracy_score(expected, predictions), 4),
              "macro F1", round(f1_score(expected, predictions, average="macro"), 4))

    old_correct = validation.words == expected
    new_correct = validation.words_and_chars == expected
    for title, mask in [("Fixed", ~old_correct & new_correct),
                        ("New errors", old_correct & ~new_correct),
                        ("Still incorrect", ~old_correct & ~new_correct)]:
        print(f"\n{title} ({mask.sum()})")
        print(validation.loc[mask, ["description", "target_category", "words", "words_and_chars"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
