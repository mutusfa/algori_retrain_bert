import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from retrain_bert.settings import PROJECT_DIR
from retrain_bert.preprocessor import main as preprocessing, load_categories


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", type=Path, default=PROJECT_DIR / "data/raw")
    parser.add_argument("--train-dir", type=Path, default=PROJECT_DIR / "data/train")
    parser.add_argument(
        "--validation-dir", type=Path, default=PROJECT_DIR / "data/validation"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Seed for random number generator"
    )
    return parser.parse_args(argv)


def clean_fuzzy_mappings(fuzzy_mappings):
    fuzzy_mappings.dropna(how="any", inplace=True)
    fuzzy_mappings.drop_duplicates(subset=["OcrValueId"], inplace=True)
    fuzzy_mappings.rename(
        columns={"OcrValue": "fuzzy_ocr", "OcrValue_Correct": "verified_ocr"},
        inplace=True,
    )
    fuzzy_mappings = fuzzy_mappings.query("fuzzy_ocr != verified_ocr")
    fuzzy_mappings = fuzzy_mappings[["fuzzy_ocr", "verified_ocr"]]
    return fuzzy_mappings


def merge_fuzzy_mappings(verified_train, fuzzy_mappings):
    fuzzy_mappings = pd.merge(
        left=verified_train,
        right=fuzzy_mappings,
        how="inner",
        on="verified_ocr",
    )
    fuzzy_mappings.rename(columns={"fuzzy_ocr": "ocr"}, inplace=True)

    verified_train["ocr"] = verified_train["verified_ocr"]

    merged = pd.concat([verified_train, fuzzy_mappings], ignore_index=True)
    merged.drop_duplicates(subset=["ocr", "category"], inplace=True, keep="first")
    assert not (merged.ocr == merged.verified_ocr).all()
    return merged


def main(args):
    args.train_dir.mkdir(parents=True, exist_ok=True)
    args.validation_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_args = argparse.Namespace(
        classified_ocrs=args.raw_data_dir / "verified_ocrs.csv",
        labels_in=None,
        labels_out=args.train_dir / "labels.csv",
        train_out=args.train_dir / "train.csv",
        categories_out=args.train_dir / "categories.csv",
    )
    preprocessing(preprocessing_args)

    verified_train = pd.read_csv(args.train_dir / "train.csv", dtype=str)
    fuzzy_mappings = pd.read_csv(args.raw_data_dir / "fuzzy_mappings.csv")
    fuzzy_mappings = clean_fuzzy_mappings(fuzzy_mappings)

    categories = load_categories()
    validation_categories = categories.sample(frac=0.05, random_state=args.random_seed)
    train_categories = categories.drop(validation_categories.index)

    new_category_verification = verified_train.query(
        "category in @validation_categories"
    )
    print(
        f"{validation_categories.shape[0]} categories chosen to check how model deals "
        "with unseen categories.\n"
        f"That's a total of {new_category_verification.shape[0]} verified ocr samples."
    )
    new_category_verification = merge_fuzzy_mappings(
        new_category_verification, fuzzy_mappings
    )
    new_category_verification.to_csv(
        args.validation_dir / "unseen_category.csv", index=False
    )

    verified_train = verified_train.query("category in @train_categories")
    train, validation = train_test_split(
        verified_train, test_size=0.05, random_state=args.random_seed
    )

    train = merge_fuzzy_mappings(train, fuzzy_mappings)
    train.to_csv(args.train_dir / "train.csv", index=False)

    validation = merge_fuzzy_mappings(validation, fuzzy_mappings)
    validation.to_csv(args.validation_dir / "validation.csv", index=False)


if __name__ == "__main__":
    args = parse_args([])
    main(args)
