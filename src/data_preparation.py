from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import RAW_DATA_DIR


@dataclass(frozen=True)
class PreparedDataset:
    name: str
    task: str
    raw_path: Path
    raw: pd.DataFrame
    cleaned: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    target: str
    positive_label: int | None = None


def find_dataset_file(candidates: Iterable[str], keyword_patterns: Iterable[str]) -> Path:
    for filename in candidates:
        path = RAW_DATA_DIR / filename
        if path.exists():
            return path

    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    for path in csv_files:
        lowered = path.name.lower()
        if all(pattern.lower() in lowered for pattern in keyword_patterns):
            return path

    expected = "\n".join(f"- {name}" for name in candidates)
    raise FileNotFoundError(
        "Dataset CSV not found in data/raw.\n"
        "Place one of these files in data/raw and run again:\n"
        f"{expected}"
    )


def _strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        mask = df[col].notna()
        df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan})
    return df


def _resolve_target(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    by_lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise ValueError(
        f"None of the expected target columns were found: {', '.join(candidates)}"
    )


def load_house_sales(path: Path) -> PreparedDataset:
    raw = pd.read_csv(path)
    df = raw.copy()
    df.columns = [col.strip() for col in df.columns]

    target = _resolve_target(df, ["price", "SalePrice", "sale_price"])
    df[target] = pd.to_numeric(df[target], errors="coerce")

    if "date" in df.columns:
        parsed_date = pd.to_datetime(df["date"], errors="coerce")
        df["sale_year"] = parsed_date.dt.year
        df["sale_month"] = parsed_date.dt.month
        df["sale_dayofweek"] = parsed_date.dt.dayofweek
        df = df.drop(columns=["date"])

    for col in ["id", "Unnamed: 0"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "zipcode" in df.columns:
        df["zipcode"] = df["zipcode"].astype(str)

    df = _strip_object_columns(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    return PreparedDataset(
        name="House Sales in King County",
        task="regression",
        raw_path=path,
        raw=raw,
        cleaned=df,
        X=X,
        y=y,
        target=target,
    )


def load_telco_churn(path: Path) -> PreparedDataset:
    raw = pd.read_csv(path)
    df = raw.copy()
    df.columns = [col.strip() for col in df.columns]

    target = _resolve_target(df, ["Churn", "churn"])

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes"})
        df["SeniorCitizen"] = df["SeniorCitizen"].replace({"0": "No", "1": "Yes"})

    df = _strip_object_columns(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=[target])

    y_text = df[target].astype(str).str.strip().str.lower()
    y = y_text.map({"no": 0, "yes": 1})
    if y.isna().any():
        bad_values = sorted(df.loc[y.isna(), target].astype(str).unique())
        raise ValueError(f"Unexpected churn labels: {bad_values}")

    df[target] = y.astype(int)
    X = df.drop(columns=[target])

    return PreparedDataset(
        name="Telco Customer Churn",
        task="classification",
        raw_path=path,
        raw=raw,
        cleaned=df,
        X=X,
        y=df[target],
        target=target,
        positive_label=1,
    )
