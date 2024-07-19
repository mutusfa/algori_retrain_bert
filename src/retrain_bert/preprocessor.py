import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from retrain_bert import settings
from retrain_bert.data.loaders import *


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.dropna(
        subset=["Category_AECOC", "Verified_OcrValueId", "OcrValue_Verified"],
        how="any",
        inplace=True,
    )
    data.drop_duplicates(
        subset=["Verified_OcrValueId", "MappingGroupId", "Category_AECOC"],
        inplace=True,
    )
    data = data.query("Category_AECOC.str.len() % 2 == 0")
    return data


def chunk_string(string, chunk_size):
    return [string[i : i + chunk_size] for i in range(0, len(string), chunk_size)]


def _split_into_categories(category_code, num_categories=5):
    categories = chunk_string(category_code, 2)
    categories += [None] * (num_categories - len(categories))
    return categories


def split_into_categories(codes: pd.DataFrame, col="Category_AECOC") -> pd.DataFrame:
    if isinstance(codes, pd.DataFrame):
        codes = codes[col]
    categories = codes.apply(_split_into_categories)
    return pd.DataFrame.from_records(categories)


def prepare_labels(categories: pd.DataFrame):
    labels = []
    for col in categories:
        labels.extend({"level": col, "cat": cat} for cat in categories[col].unique())
        labels.append({"level": col, "cat": "UNKNOWN"})
    labels = pd.DataFrame.from_records(labels).dropna().sort_values(["level", "cat"])
    labels.reset_index(inplace=True, drop=True)
    labels["level"] += 1
    labels["label"] = labels.index.copy()
    labels.set_index(["level", "cat"], inplace=True)
    return labels


def encode_label(label_ids, labels_table):
    one_hot = np.zeros(len(labels_table))
    one_hot[label_ids] = 1
    return one_hot


def get_labels_conf(labels: pd.DataFrame) -> list:
    labels_conf = []
    level_start = 0
    level_end = 0
    for level in range(settings.DEEPEST_LEVEL):
        level_end = level_start + len(labels.loc[level + 1])
        labels_conf.append(
            {
                "level": level + 1,
                "start": level_start,
                "end": level_end,
                "num_classes": len(labels.loc[level + 1]),
            }
        )
        level_start = level_end
    return labels_conf


def to_label_id(
    category_code: str, labels: pd.DataFrame, deepest_level: int = 5
) -> int:
    category_code_list = [
        category_code[2 * i : 2 * i + 2] for i in range(len(category_code) // 2)
    ]
    category_code_list += ["UNKNOWN"] * (deepest_level - len(category_code_list))
    return [
        labels.loc[(level + 1, cat), "label"]
        for level, cat in enumerate(category_code_list)
    ]


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--classified-ocrs", type=Path, required=True)
    parser.add_argument("--labels-in", type=Path)
    parser.add_argument("--labels-out", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--categories-out", type=Path, required=True)
    return parser.parse_args(args)


def main(args):
    data = load_raw_data(args.classified_ocrs)
    print(f"Loaded {len(data)} classified OCRs")
    data = clean_data(data)
    print(f"Cleaned data, {len(data)} classified OCRs left")
    categories = split_into_categories(data)
    labels = prepare_labels(categories)
    labels.to_csv(args.labels_out)
    if args.labels_in:
        labels = load_labels(args.labels_in)
    labels_conf = get_labels_conf(labels)
    label_ids = data["Category_AECOC"].apply(to_label_id, labels=labels)
    label_ids = pd.DataFrame.from_records(
        label_ids, columns=[f"level_{l+1}" for l in range(settings.DEEPEST_LEVEL)]
    ).astype(int)
    for level, col in zip(range(settings.DEEPEST_LEVEL), label_ids.columns):
        label_ids[col] -= labels_conf[level]["start"]
    label_ids.index = data.index
    data = pd.concat([data, label_ids], axis=1)
    data.rename(
        columns={"OcrValue_Verified": "verified_ocr", "Category_AECOC": "category"},
        inplace=True,
    )
    data[["verified_ocr", "category"] + label_ids.columns.to_list()].to_csv(
        args.train_out, index=False
    )
    print(f"Saved {len(data)} classified OCRs as training data")
    data.category.drop_duplicates().sort_values().to_csv(
        args.categories_out, index=False, header=False
    )


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
