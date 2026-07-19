from __future__ import annotations

import argparse

from .config import (
    CV_SPLITS,
    HOUSE_DATA_CANDIDATES,
    HOUSE_MODEL_PATH,
    TELCO_DATA_CANDIDATES,
    TELCO_MODEL_PATH,
    ensure_project_dirs,
)
from .data_preparation import find_dataset_file, load_house_sales, load_telco_churn
from .modeling import evaluate_dataset, save_artifact
from .reporting import save_dataset_reports, save_model_results, write_project_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML models for the project.")
    parser.add_argument(
        "--cv",
        type=int,
        default=CV_SPLITS,
        help="Number of cross-validation folds. Default: 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    try:
        house_path = find_dataset_file(HOUSE_DATA_CANDIDATES, ["house"])
        telco_path = find_dataset_file(TELCO_DATA_CANDIDATES, ["churn"])
    except FileNotFoundError as exc:
        print(str(exc))
        print("\nDownload the CSV files from the Kaggle links in Datasets.txt.")
        raise SystemExit(1) from None

    house_dataset = load_house_sales(house_path)
    telco_dataset = load_telco_churn(telco_path)

    print(f"Loaded {house_dataset.name}: {house_dataset.cleaned.shape}")
    print(f"Loaded {telco_dataset.name}: {telco_dataset.cleaned.shape}")

    save_dataset_reports(house_dataset)
    save_dataset_reports(telco_dataset)

    print("Training and comparing regression models...")
    house_eval = evaluate_dataset(house_dataset, cv_splits=args.cv)
    save_model_results("house_price_regression", house_eval.results)
    save_artifact(house_eval.best_artifact, HOUSE_MODEL_PATH)

    print("Training and comparing classification models...")
    telco_eval = evaluate_dataset(telco_dataset, cv_splits=args.cv)
    save_model_results("telco_churn_classification", telco_eval.results)
    save_artifact(telco_eval.best_artifact, TELCO_MODEL_PATH)

    summary_path = write_project_summary(
        house_dataset=house_dataset,
        telco_dataset=telco_dataset,
        house_results=house_eval.results,
        telco_results=telco_eval.results,
        house_best=house_eval.best_row,
        telco_best=telco_eval.best_row,
    )

    print("\nDone.")
    print(f"Regression best model: {house_eval.best_row['model_name']}")
    print(f"Classification best model: {telco_eval.best_row['model_name']}")
    print(f"Saved regression model: {HOUSE_MODEL_PATH}")
    print(f"Saved classification model: {TELCO_MODEL_PATH}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
