from pathlib import Path
from sre_parse import CATEGORIES

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEEPEST_LEVEL = 5
MAX_SEQUENCE_LENGTH = 32
BACKBONE_MODEL = "bert-base-multilingual-uncased"

DATA_DIR = PROJECT_DIR / "data"
INFERENCE_MODEL_PATH = PROJECT_DIR / "models/bert_finetuned.keras"
CATEGORIES_PATH = PROJECT_DIR / "data/categories.csv"
