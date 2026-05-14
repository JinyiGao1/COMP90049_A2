from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

DIABETIC_DATA_PATH = RAW_DATA_DIR / "diabetic_data.csv"
IDS_MAPPING_PATH = RAW_DATA_DIR / "IDS_mapping.csv"
