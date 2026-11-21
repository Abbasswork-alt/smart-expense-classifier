from pathlib import Path
import joblib

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "expense_classifier_final.joblib"
)

_model = None


def predict(text):
    global _model

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a non-empty description.")

    if _model is None:
        _model = joblib.load(MODEL_PATH)

    probabilities = _model.predict_proba([text.strip()])[0]
    best_index = probabilities.argmax()

    return {
        "category": str(_model.classes_[best_index]),
        "confidence": float(probabilities[best_index])
    }