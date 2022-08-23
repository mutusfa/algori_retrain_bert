import json
from pathlib import Path

import pandas as pd

from retrain_bert.settings import DEEPEST_LEVEL, PROJECT_DIR


def load_raw_data(path: Path = None) -> pd.DataFrame:
    path = path or PROJECT_DIR / "data/raw/classified_ocr_2022_08_11.csv"
    return pd.read_csv(
        path,
        dtype={"Category_MasterProduct": str, "Category_BERT": str},
    )


def clean_data(data: pd.DataFrame, inplace=True) -> pd.DataFrame:
    data.dropna(subset=["Category_MasterProduct", "OcrValueId"], inplace=inplace)
    data.drop_duplicates(
        subset=["OcrValueId", "MappingGroupId", "Category_MasterProduct"], inplace=True
    )


def chunk_string(string, chunk_size):
    return [string[i : i + chunk_size] for i in range(0, len(string), chunk_size)]


def _split_into_categories(category_code, num_categories=5):
    categories = chunk_string(category_code, 2)
    categories += [None] * (num_categories - len(categories))
    return categories


def split_into_categories(
    df: pd.DataFrame, col="Category_MasterProduct"
) -> pd.DataFrame:
    codes = df[col]
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


def load_labels(path: Path = None) -> pd.DataFrame:
    path = path or PROJECT_DIR / "data/labels.csv"
    labels = pd.read_csv(path)
    labels.set_index(["level", "cat"], inplace=True)
    return labels


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


if __name__ == "__main__":
    data = load_raw_data()
    clean_data(data)
    categories = split_into_categories(data)
    labels = prepare_labels(categories)
    labels.to_csv(PROJECT_DIR / "data/labels.csv")
    label_ids = data["Category_MasterProduct"].apply(to_label_id, labels=labels)
    label_ids = pd.DataFrame.from_records(
        label_ids, columns=[f"level_{l+1}" for l in range(DEEPEST_LEVEL)]
    ).astype(int)
    label_ids.index = data.index
    data = pd.concat([data, label_ids], axis=1)
    data[["OcrValue"] + label_ids.columns.to_list()].to_csv(PROJECT_DIR / "data/train/train.csv", index=False)

    with open(PROJECT_DIR / "data/raw/missing_ocr_values.json") as mo:
        with open(PROJECT_DIR / "data/train/ocr_values.txt", "w") as f:
            ocr_in_db = pd.read_csv(PROJECT_DIR / "data/raw/OCR_Values.csv")
            missing_ocr = json.load(mo)
            ocrs = ocr_in_db.Name.to_list() + missing_ocr
            ocrs = [str(o).strip().upper() for o in ocrs if o]
            f.write("\n".join(ocrs))
