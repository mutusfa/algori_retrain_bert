import argparse
from pathlib import Path
import sys

import pandas as pd

from retrain_bert import settings


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    return parser.parse_args(args)


def main(args):
    raw_table = pd.read_csv(args.input, dtype=str)
    code_name_mappings = []
    for level in range(1, settings.DEEPEST_LEVEL + 1):
        code = raw_table[f"Level{level}Code"].rename("code")
        names = raw_table[[f"Level{l}Name" for l in range(1, level + 1)]]
        name = names.apply(
            lambda x: x.str.cat(sep=" > ", na_rep=""), axis="columns"
        ).rename("name")
        mappings = pd.concat([code, name], axis=1)
        mappings.drop_duplicates(inplace=True, subset=["code"])
        code_name_mappings.append(mappings)
    code_name_mappings = (
        pd.concat(code_name_mappings)
        .dropna()
        .reset_index(drop=True)
        .sort_values("code")
    )
    code_name_mappings = code_name_mappings.query("code.str.len() % 2 == 0")
    code_name_mappings.to_csv(args.output, index=False)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
