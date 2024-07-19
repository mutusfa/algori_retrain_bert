import argparse
import logging
import os
from pathlib import Path
import sys

import dotenv
import pandas as pd
import sqlalchemy as sa

from retrain_bert.settings import PROJECT_DIR

dotenv.load_dotenv(PROJECT_DIR / ".env")

LOG = logging.getLogger(__name__)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "data/raw")
    parser.add_argument(
        "--queries-dir", type=Path, default=PROJECT_DIR / "data/queries"
    )
    return parser.parse_args(argv)


def connection():
    conn = sa.create_engine(os.getenv("ELASTIC_DATABASE_CONNECTION_STRING"))
    return conn


def execute_query(query_pth: Path, output_dir: Path, conn):
    with open(query_pth, "r") as f:
        query = f.read()
    print(f"Executing query: {query_pth.stem}")
    df = pd.read_sql(query, conn)
    df.to_csv(output_dir / f"{query_pth.stem}.csv", index=False)


def main(args):
    conn = connection()
    for query_pth in args.queries_dir.rglob("*.sql"):
        execute_query(query_pth, args.output_dir, conn)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    sys.exit(main(args))
