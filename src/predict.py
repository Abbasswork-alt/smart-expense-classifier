from pathlib import Path
from collections import Counter
from functools import lru_cache
import joblib
from src.text_processing import clean_text

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "expense_classifier_final.joblib"
)

_model = None


class UnrecognizedDescription(ValueError):
    """The description has no features in any fitted vocabulary."""


@lru_cache(maxsize=1)
def _notes_model():
    from src.semantic_encoder import Encoder
    folder = MODEL_PATH.parent / "semantic_encoder"
    return Encoder(folder), joblib.load(folder / "english_notes_head.joblib")


def _bank_statement(text):
    return text.lstrip().lower().startswith(("[debit]", "[credit]"))


def _scores(descriptions):
    base = _model.predict_proba(descriptions)
    if _bank_statement(descriptions[0]):
        return base
    encoder, head = _notes_model()
    return (base + head.predict_proba(encoder.encode(descriptions))) / 2


def predict(text):
    global _model

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a non-empty description.")

    if _model is None:
        _model = joblib.load(MODEL_PATH)

    description = [text.strip()]
    if all(
        fold.estimator.named_steps["tfidf"].transform(description).nnz == 0
        for fold in _model.calibrated_classifiers_
    ):
        raise UnrecognizedDescription("Add what you bought or paid for so the model can classify it.")

    probabilities = _scores(description)[0]
    best_index = probabilities.argmax()

    return {
        "category": str(_model.classes_[best_index]),
        "confidence": float(probabilities[best_index])
    }


def explain(text):
    """Return the prediction and up to three positive contributions to its SVM score."""
    result = predict(text)
    if not _bank_statement(text):
        tokens = clean_text(text).split()
        # Remove each distinct word and measure the change for the winning class.
        words = list(dict.fromkeys(tokens))
        omissions = [" ".join(t for t in tokens if t != word) for word in words]
        index = list(_model.classes_).index(result["category"])
        drops = result["confidence"] - _scores(omissions)[:, index]
        ranked = sorted(zip(words, drops), key=lambda item: -item[1])
        result["features"] = [{"feature": w, "contribution": float(d)}
                              for w, d in ranked[:3] if d > 0]
        return result
    totals = Counter()
    folds = _model.calibrated_classifiers_
    for fold in folds:
        vectorizer = fold.estimator.named_steps["tfidf"]
        classifier = fold.estimator.named_steps["classifier"]
        row = vectorizer.transform([text.strip()]).tocsr()
        names = vectorizer.get_feature_names_out()
        class_index = list(classifier.classes_).index(result["category"])
        weights = classifier.coef_[class_index]
        for index, value in zip(row.indices, row.data):
            totals[str(names[index])] += value * weights[index]

    result["features"] = [
        {"feature": name, "contribution": float(total / len(folds))}
        for name, total in totals.most_common(3)
        if total > 0
    ]
    return result
