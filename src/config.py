from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

HOUSE_MODEL_PATH = MODELS_DIR / "house_price_best_model.joblib"
TELCO_MODEL_PATH = MODELS_DIR / "telco_churn_best_model.joblib"

HOUSE_DATA_CANDIDATES = [
    "kc_house_data.csv",
    "house_data.csv",
    "housesalesprediction.csv",
    "king_county_house_sales.csv",
    "House Sales in King County, USA.csv",
]

TELCO_DATA_CANDIDATES = [
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "telco_customer_churn.csv",
    "Telco-Customer-Churn.csv",
    "Telco Customer Churn.csv",
]


def ensure_project_dirs() -> None:
    for path in [
        RAW_DATA_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        OUTPUTS_DIR,
        MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
