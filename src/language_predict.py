from functools import lru_cache

import joblib

from src.predict import MODEL_PATH, UnrecognizedDescription, explain


@lru_cache(maxsize=2)
def load_language(language):
    return joblib.load(MODEL_PATH.parent / f"{language}.joblib")


def explain_language(text, language):
    if language == "english":
        return explain(text)
    if language not in {"arabic", "mixed"}:
        raise ValueError("Unknown language page")
    model = load_language(language)
    pipeline = model if language == "mixed" else model.calibrated_classifiers_[0].estimator
    features = pipeline.named_steps["features"]
    row = features.transform([text])
    if not row.nnz:
        raise UnrecognizedDescription("أضف وصفاً لما اشتريته أو دفعت ثمنه.")
    probs = model.predict_proba([text])[0]
    index = probs.argmax()
    weights = pipeline.named_steps["classifier"].coef_[index]
    words = features.transformer_list[0][1].get_feature_names_out()
    ranked = sorted(((words[i], float(v * weights[i])) for i, v in zip(row.indices, row.data)
                     if i < len(words) and v * weights[i] > 0), key=lambda item: -item[1])
    return {"category": str(model.classes_[index]), "confidence": float(probs[index]),
            "features": [{"feature": word, "contribution": score} for word, score in ranked[:3]]}
