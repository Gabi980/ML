from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, REPORTS_DIR, TABLES_DIR
from .data_preparation import PreparedDataset


def _safe_filename(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(",", "")
        .replace("__", "_")
    )


def save_dataset_reports(dataset: PreparedDataset) -> None:
    slug = _safe_filename(dataset.name)
    df = dataset.cleaned

    df.describe(include="all").T.to_csv(TABLES_DIR / f"{slug}_statistical_summary.csv")
    df.isna().sum().sort_values(ascending=False).to_csv(
        TABLES_DIR / f"{slug}_missing_values.csv", header=["missing_values"]
    )
    df.dtypes.astype(str).to_csv(TABLES_DIR / f"{slug}_dtypes.csv", header=["dtype"])

    if dataset.task == "regression":
        _plot_regression_eda(dataset, slug)
    else:
        _plot_classification_eda(dataset, slug)


def _plot_regression_eda(dataset: PreparedDataset, slug: str) -> None:
    df = dataset.cleaned
    target = dataset.target

    plt.figure(figsize=(8, 5))
    sns.histplot(df[target], kde=True, bins=40)
    plt.title("Target distribution: house price")
    plt.xlabel(target)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{slug}_target_distribution.png", dpi=160)
    plt.close()

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr(numeric_only=True)
        plt.figure(figsize=(11, 9))
        sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
        plt.title("Correlation matrix")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{slug}_correlation_matrix.png", dpi=160)
        plt.close()

        target_corr = (
            corr[target]
            .drop(labels=[target], errors="ignore")
            .sort_values(key=lambda s: s.abs(), ascending=False)
            .head(12)
        )
        target_corr.to_csv(TABLES_DIR / f"{slug}_top_target_correlations.csv")

        plt.figure(figsize=(8, 5))
        target_corr.sort_values().plot(kind="barh")
        plt.title("Top correlations with target")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{slug}_top_target_correlations.png", dpi=160)
        plt.close()

    if "sqft_living" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df.sample(min(len(df), 3000), random_state=42), x="sqft_living", y=target, alpha=0.35)
        plt.title("Price vs living area")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{slug}_sqft_living_vs_price.png", dpi=160)
        plt.close()


def _plot_classification_eda(dataset: PreparedDataset, slug: str) -> None:
    df = dataset.cleaned
    target = dataset.target

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=target)
    plt.title("Target distribution: churn")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{slug}_target_distribution.png", dpi=160)
    plt.close()

    numeric_cols = [
        col for col in ["tenure", "MonthlyCharges", "TotalCharges"] if col in df.columns
    ]
    for col in numeric_cols:
        plt.figure(figsize=(7, 4))
        sns.histplot(data=df, x=col, hue=target, kde=True, bins=30, element="step")
        plt.title(f"{col} distribution by churn")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{slug}_{col.lower()}_by_churn.png", dpi=160)
        plt.close()

    for col in ["Contract", "PaymentMethod", "InternetService"]:
        if col in df.columns:
            churn_rate = df.groupby(col)[target].mean().sort_values(ascending=False)
            churn_rate.to_csv(TABLES_DIR / f"{slug}_{col.lower()}_churn_rate.csv")

            plt.figure(figsize=(8, 4))
            churn_rate.plot(kind="bar")
            plt.title(f"Churn rate by {col}")
            plt.ylabel("Churn rate")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"{slug}_{col.lower()}_churn_rate.png", dpi=160)
            plt.close()


def save_model_results(task_name: str, results: pd.DataFrame) -> Path:
    path = TABLES_DIR / f"{task_name}_model_comparison.csv"
    results.to_csv(path, index=False)
    return path


def write_project_summary(
    house_dataset: PreparedDataset,
    telco_dataset: PreparedDataset,
    house_results: pd.DataFrame,
    telco_results: pd.DataFrame,
    house_best: dict,
    telco_best: dict,
) -> Path:
    path = REPORTS_DIR / "project_summary.md"

    house_top = house_results.sort_values("holdout_rmse").head(3)
    telco_top = telco_results.sort_values("holdout_f1", ascending=False).head(3)

    content = f"""# Machine Learning Project Summary

## Tema

Platforma de predictii de business folosind Machine Learning:
- regresie pentru estimarea pretului locuintelor din King County;
- clasificare pentru predictia churn-ului clientilor telecom.

## Dataseturi

### {house_dataset.name}
- Fisier: `{house_dataset.raw_path.name}`
- Task: regresie
- Target: `{house_dataset.target}`
- Dimensiune raw: {house_dataset.raw.shape[0]} randuri x {house_dataset.raw.shape[1]} coloane
- Dimensiune dupa curatare: {house_dataset.cleaned.shape[0]} randuri x {house_dataset.cleaned.shape[1]} coloane

### {telco_dataset.name}
- Fisier: `{telco_dataset.raw_path.name}`
- Task: clasificare binara
- Target: `{telco_dataset.target}` (`No` = 0, `Yes` = 1)
- Dimensiune raw: {telco_dataset.raw.shape[0]} randuri x {telco_dataset.raw.shape[1]} coloane
- Dimensiune dupa curatare: {telco_dataset.cleaned.shape[0]} randuri x {telco_dataset.cleaned.shape[1]} coloane

## Transformari aplicate

- eliminare duplicate;
- conversie coloane numerice stocate ca text, de exemplu `TotalCharges`;
- extragere componente temporale din `date` pentru datasetul de case;
- eliminare identificatori fara valoare predictiva (`id`, `customerID`);
- imputare valori lipsa cu mediana pentru variabile numerice;
- imputare valori lipsa cu moda pentru variabile categorice;
- standardizare variabile numerice;
- one-hot encoding pentru variabile categorice;
- impartire train/test si evaluare prin cross-validation.

## Cele mai bune modele

### Regresie
- Model selectat: **{house_best['model_name']}**
- RMSE holdout: **{house_best['holdout_rmse']:.4f}**
- MAE holdout: **{house_best['holdout_mae']:.4f}**
- R2 holdout: **{house_best['holdout_r2']:.4f}**

Top 3 modele dupa RMSE:

{house_top.to_markdown(index=False)}

### Clasificare
- Model selectat: **{telco_best['model_name']}**
- F1 holdout: **{telco_best['holdout_f1']:.4f}**
- Accuracy holdout: **{telco_best['holdout_accuracy']:.4f}**
- ROC AUC holdout: **{telco_best['holdout_roc_auc']:.4f}**

Top 3 modele dupa F1:

{telco_top.to_markdown(index=False)}

## Artefacte generate

- `reports/tables`: sumar statistic, valori lipsa, comparatii modele;
- `reports/figures`: vizualizari EDA;
- `outputs/models`: modelele finale salvate cu `joblib`;
- `app.py`: interfata minima Streamlit pentru predictii.
"""
    path.write_text(content, encoding="utf-8")

    metadata_path = REPORTS_DIR / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "house_best": house_best,
                "telco_best": telco_best,
                "house_dataset": str(house_dataset.raw_path),
                "telco_dataset": str(telco_dataset.raw_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return path
