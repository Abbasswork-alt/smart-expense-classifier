"""Run with the project's dependencies: python tests/audit_classifier.py"""

import re
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import bank_labels, everyday_labels, model_splits, remove_bank_conflicts
from src.predict import MODEL_PATH, UnrecognizedDescription, predict
from src.text_processing import clean_text


# Four examples per category, labeled before running predictions.
PROBES = {
    "Bills": ["paid my electricity bill", "broadband for the apartment",
             "monthly mobile phone plan", "water bill for October"],
    "Entertainment": ["two cinema tickets", "video game download",
                      "concert admission", "renewed my movie streaming subscription"],
    "Food": ["bread and eggs from the corner shop", "vegetables for dinner",
             "sandwich at the cafe", "bought milk and cheese"],
    "Health": ["prescription refill", "annual dental checkup",
               "physiotherapy appointment", "antibiotics from the pharmacy"],
    "Other": ["sent money to Sam", "monthly apartment rent",
              "passport renewal fee", "donation to charity"],
    "Shopping": ["new running shoes", "bought a desk lamp",
                 "replacement laptop charger", "winter coat purchase"],
    "Transport": ["taxi fare home", "bus pass for the week",
                  "parking at the station", "petrol for the car"],
}

# Hold out these literal brand names (two per intended category). Other aliases
# can remain. Score the very same matching test rows before and after removal.
BRANDS = ["eversource", "boost mobile", "netflix", "spotify", "starbucks",
          "whole foods", "comfort dental", "providence health", "cigna",
          "marriott", "target", "ikea", "uber", "hertz"]

DEMO_LABELS = {
    "Carrefour groceries": "Food", "Starbucks coffee": "Food", "McDonalds lunch": "Food",
    "Whole Foods groceries": "Food", "Uber ride to airport": "Transport",
    "Shell fuel payment": "Transport", "Delta Air Lines ticket": "Transport",
    "Water utility payment": "Bills", "Internet bill Comcast": "Bills",
    "Netflix monthly payment": "Entertainment", "Spotify subscription": "Entertainment",
    "Disney+ subscription": "Entertainment", "Amazon headphones purchase": "Shopping",
    "Walmart household shopping": "Shopping", "Target clothing purchase": "Shopping",
    "Dental clinic appointment": "Health", "Doctor medical visit": "Health",
    "Monthly rent payment": "Other", "Microsoft 365 subscription": "Other",
    "Hilton hotel booking": "Other",
}


def fit(template, train):
    folds = list(GroupKFold(3).split(train, groups=train.key))
    model = clone(template).set_params(cv=folds)
    return model.fit(train.description, train.target_category)


def score(model, frame):
    guess = model.predict(frame.description)
    return (round(accuracy_score(frame.target_category, guess), 5),
            round(f1_score(frame.target_category, guess, labels=model.classes_,
                           average="macro", zero_division=0), 5))


def check_demo():
    app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert app.warning and not app.metric
    options = set(app.selectbox[0].proto.options[1:])
    assert options == set(DEMO_LABELS)
    correct = 0
    for description, expected in DEMO_LABELS.items():
        app.selectbox[0].select(description).run()
        assert app.text_input[0].value == description and not app.metric
        app.button[0].click().run()
        actual = predict(description)
        assert not app.exception and actual["category"] in [s.value for s in app.subheader]
        assert app.metric[0].value == f"{actual['confidence']:.1%}"
        correct += actual["category"] == expected
        print("demo", description, expected, actual)
    app.text_input[0].set_value("taxi fare home").run()
    assert not app.metric
    app.button[0].click().run()
    assert not app.exception and app.metric
    app.text_input[0].set_value("!!!").run()
    app.button[0].click().run()
    assert app.warning and not app.metric and not app.exception
    print("demo correct", correct, "/", len(DEMO_LABELS))


def main():
    model = joblib.load(MODEL_PATH)
    data = bank_labels()
    notes = everyday_labels(data)
    assert not data.isna().any().any() and set(data.target_category) == set(PROBES)
    parking = data.loc[data.key.str.contains(r"\bparking\b")]
    print("parking label audit", parking.groupby(["category", "target_category"]).size().to_dict())
    airport_parking = parking.loc[parking.key.str.contains(r"\bairport parking\b")]
    assert len(airport_parking) and airport_parking.target_category.eq("Transport").all()
    conflicts = data.groupby("key").target_category.nunique().gt(1)
    clean = remove_bank_conflicts(data)
    reserved = {clean_text(text) for phrases in PROBES.values() for text in phrases}
    reserved.update(map(clean_text, DEMO_LABELS))
    assert not reserved.intersection(notes.key)
    print("prepared rows, conflict groups and rows, usable rows and groups",
          len(data), conflicts.sum(), len(data) - len(clean), len(clean), clean.key.nunique())
    for seed in [42, 53, 64]:
        train, validation, test = model_splits(clean, notes, seed)
        assert not (set(train.key) & set(test.key) or set(train.key) & set(validation.key)
                    or set(validation.key) & set(test.key))
        refit = fit(model, train)
        print("seed", seed, "sizes", len(train), len(validation), len(test),
              "train/val/test (accuracy, macro F1)",
              [score(refit, part) for part in [train, validation, test]])
        if seed != 42:
            continue
        print("shipped artifact scores", [score(model, p) for p in [train, validation, test]])
        for source, rows in test.groupby("source"):
            print("test results", source, len(rows), score(model, rows))
        print("saved vs refit test probability max absolute difference",
              abs(model.predict_proba(test.description) - refit.predict_proba(test.description)).max())
        guess = model.predict(test.description)
        print("majority baseline", train.target_category.mode()[0],
              (test.target_category == train.target_category.mode()[0]).mean())
        print(classification_report(test.target_category, guess, digits=4))
        print("confusion matrix", model.classes_, confusion_matrix(
            test.target_category, guess, labels=model.classes_), sep="\n")
        pattern = r"\b(?:" + "|".join(map(re.escape, BRANDS)) + r")\b"
        retained = train.loc[~train.key.str.contains(pattern)]
        brand_test = test.loc[test.source.eq("bank") & test.key.str.contains(pattern)]
        held_out = fit(model, retained)
        size_control = fit(model, train.sample(n=len(retained), random_state=42).sort_index())
        print("brand holdout train removed, test support", len(train)-len(retained),
              brand_test.target_category.value_counts().to_dict())
        print("same brand rows before/after exclusion", score(refit, brand_test), score(held_out, brand_test))
        print("same brand rows after dropping an equal number of random rows", score(size_control, brand_test))
        for word in ["uber", "ride", "airport", "carrefour", "groceries"]:
            counts = train.loc[train.key.str.contains(r"\b" + word + r"\b")].target_category.value_counts()
            print("train rows containing", word, counts.to_dict())

    expected, guesses = [], []
    known = {key.removeprefix("debit ") for key in clean.key}
    vectorizers = [fold.estimator.named_steps["tfidf"] for fold in model.calibrated_classifiers_]
    no_features = 0
    for category, descriptions in PROBES.items():
        for description in descriptions:
            assert clean_text(description) not in known
            try:
                result = predict(description)
            except UnrecognizedDescription:
                result = {"category": "unclassified", "confidence": None}
            no_features += all(v.transform([description]).nnz == 0 for v in vectorizers)
            guesses.append(result["category"])
            expected.append(category)
            print("probe", description, category, result)
    print("probe correct and macro F1", np.sum(np.equal(expected, guesses)),
          len(expected), f1_score(expected, guesses, labels=model.classes_, average="macro"))
    print("probes without any learned features", no_features)
    for description in ["Uber ride to airport", "Carrefour groceries"]:
        for index, vectorizer in enumerate(vectorizers):
            indices = vectorizer.transform([description]).indices
            print("recognized features", description, index, vectorizer.get_feature_names_out()[indices])
    for variant in ["Uber ride", "airport", "Uber trip"]:
        print("wording check", variant, predict(variant))
    for invalid in [None, "", "   ", 123]:
        try:
            predict(invalid)
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid input rejection: {invalid!r}")
    for value in ["12345", "!!!", "qxzv"]:
        try:
            predict(value)
        except UnrecognizedDescription:
            continue
        raise AssertionError(f"Expected a request for more detail: {value!r}")
    check_demo()


if __name__ == "__main__":
    main()
