from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.text_processing import clean_text


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {"Bills", "Entertainment", "Food", "Health", "Other", "Shopping", "Transport"}
EXTRA = ["Housing", "Insurance", "Personal Care", "Education", "Fees"]


def bank_labels():
    bank = pd.read_csv(ROOT / "data/raw/processed/expenses_labeled_v1.csv")
    bank["target_category"] = bank.target_category.replace(dict.fromkeys(EXTRA, "Other"))
    bank["key"] = bank.description.map(clean_text)
    bank["source"] = "bank"
    return bank


def everyday_labels(bank):
    notes = pd.read_csv(ROOT / "data/everyday_expenses.csv")
    if notes.isna().any().any() or set(notes.target_category) != CATEGORIES:
        raise ValueError("Every everyday example needs a description and one of the seven labels.")
    if set(notes.split) != {"train", "validation", "test"}:
        raise ValueError("Expected fixed train, validation and test partitions.")
    notes["key"] = notes.description.map(clean_text)
    note_keys = notes.key.str.removeprefix("debit ")
    bank_keys = set(bank.key.str.removeprefix("debit "))
    if note_keys.eq("").any() or note_keys.duplicated().any() or note_keys.isin(bank_keys).any():
        raise ValueError("Review the empty or repeated expense descriptions before training.")
    notes["source"] = "everyday"
    return notes


def remove_bank_conflicts(bank):
    count = bank.groupby("key").target_category.transform("nunique")
    return bank.loc[count == 1].copy()


def group_split(frame, seed, fraction):
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    left, right = next(splitter.split(frame, groups=frame.key))
    return frame.iloc[left], frame.iloc[right]


def model_splits(bank, notes, seed=42):
    development, test = group_split(bank, seed, 0.20)
    train, validation = group_split(development, seed + 1, 0.25)
    partitions = []
    for name, bank_rows in [("train", train), ("validation", validation), ("test", test)]:
        everyday_rows = notes.loc[notes.split == name]
        columns = ["description", "target_category", "key", "source"]
        partitions.append(pd.concat([bank_rows[columns], everyday_rows[columns]], ignore_index=True))
    return tuple(partitions)
