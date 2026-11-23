"""Train the Arabic and mixed classifiers and report their partition scores."""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.text_processing import clean_text


def train_language(language):
    data = pd.read_csv(ROOT / f"data/{language}_expenses.csv")
    assert data.description.map(clean_text).is_unique
    train = data[data.split == "train"]
    groups = train.description
    if language == "arabic":
        sources = pd.read_csv(ROOT / "data/arabic_expense_sources.csv")
        parents = sources[sources.origin == "arabic-paraphrase"].set_index("description").reference
        groups = groups.map(parents).fillna(groups)
    features = FeatureUnion([
        ("words", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2))),
        ("chars", TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb", ngram_range=(3, 5))),
    ])
    pipe = Pipeline([("features", features), ("classifier", LinearSVC(random_state=42))])
    if language == "arabic":
        pipe.set_params(features__transformer_weights={"words": 0.5, "chars": 1}, classifier__C=2)
    cv = StratifiedGroupKFold(3, shuffle=True, random_state=42)
    folds = list(cv.split(train.description, train.target_category, groups))
    model = CalibratedClassifierCV(pipe, cv=folds, ensemble=False)
    if language == "mixed":
        # Direct multinomial probabilities avoid fitting seven extra calibration curves on 84 rows.
        model = pipe.set_params(classifier=LogisticRegression(C=2, max_iter=1000, random_state=42))
    model.fit(train.description, train.target_category)
    for split, rows in data.groupby("split"):
        pred = model.predict(rows.description)
        print(language, split, len(rows), accuracy_score(rows.target_category, pred),
              f1_score(rows.target_category, pred, average="macro"))
    joblib.dump(model, ROOT / f"models/{language}.joblib", compress=3)


if __name__ == "__main__":
    for language in ["arabic", "mixed"]:
        train_language(language)
