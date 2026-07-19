from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .config import RANDOM_STATE, TEST_SIZE
from .config import TABLES_DIR
from .data_preparation import PreparedDataset


@dataclass
class EvaluationOutput:
    results: pd.DataFrame
    best_row: dict[str, Any]
    best_pipeline: Pipeline
    best_artifact: dict[str, Any]


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [col for col in X.columns if is_numeric_dtype(X[col])]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def regression_models() -> list[tuple[str, str, Any]]:
    return [
        ("base", "Linear Regression", LinearRegression()),
        ("base", "Ridge Regression", Ridge()),
        ("base", "Lasso Regression", Lasso(max_iter=10000, random_state=RANDOM_STATE)),
        ("base", "KNN Regressor", KNeighborsRegressor(n_neighbors=7)),
        (
            "base",
            "Decision Tree Regressor",
            DecisionTreeRegressor(random_state=RANDOM_STATE),
        ),
        (
            "ensemble",
            "Random Forest Regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "ensemble",
            "AdaBoost Regressor",
            AdaBoostRegressor(n_estimators=100, random_state=RANDOM_STATE),
        ),
    ]


def classification_models() -> list[tuple[str, str, Any]]:
    return [
        (
            "base",
            "Logistic Regression",
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        ),
        ("base", "KNN Classifier", KNeighborsClassifier(n_neighbors=7)),
        ("base", "Gaussian Naive Bayes", GaussianNB()),
        (
            "base",
            "Decision Tree Classifier",
            DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        ),
        (
            "base",
            "Support Vector Machine",
            SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE),
        ),
        (
            "ensemble",
            "Random Forest Classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        (
            "ensemble",
            "AdaBoost Classifier",
            AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
        ),
    ]


def build_feature_schema(X: pd.DataFrame) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for col in X.columns:
        series = X[col]
        if is_numeric_dtype(series):
            clean = pd.to_numeric(series, errors="coerce").dropna()
            if clean.empty:
                default = 0.0
                min_value = None
                max_value = None
            else:
                default = float(clean.median())
                min_value = float(clean.min())
                max_value = float(clean.max())
            schema[col] = {
                "kind": "numeric",
                "default": default,
                "min": min_value,
                "max": max_value,
            }
        else:
            clean = series.dropna().astype(str)
            if clean.empty:
                choices = [""]
                default = ""
            else:
                choices = sorted(clean.unique().tolist())
                default = clean.mode().iloc[0]
            schema[col] = {
                "kind": "categorical",
                "choices": choices,
                "default": default,
            }
    return schema


def evaluate_dataset(dataset: PreparedDataset, cv_splits: int) -> EvaluationOutput:
    if dataset.task == "regression":
        return _evaluate_regression(dataset, cv_splits)
    return _evaluate_classification(dataset, cv_splits)


def _evaluate_regression(dataset: PreparedDataset, cv_splits: int) -> EvaluationOutput:
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    }

    rows: list[dict[str, Any]] = []
    fitted_pipelines: dict[str, Pipeline] = {}

    for group, name, estimator in regression_models():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )

        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        holdout_mse = mean_squared_error(y_test, y_pred)
        row = {
            "model_group": group,
            "model_name": name,
            "cv_mae": -float(np.mean(cv_scores["test_mae"])),
            "cv_rmse": float(np.sqrt(-np.mean(cv_scores["test_mse"]))),
            "cv_r2": float(np.mean(cv_scores["test_r2"])),
            "holdout_mae": float(mean_absolute_error(y_test, y_pred)),
            "holdout_rmse": float(np.sqrt(holdout_mse)),
            "holdout_r2": float(r2_score(y_test, y_pred)),
        }
        rows.append(row)
        fitted_pipelines[name] = pipeline

    results = pd.DataFrame(rows).sort_values("holdout_rmse")
    best_row = results.iloc[0].to_dict()
    best_pipeline = fitted_pipelines[best_row["model_name"]]

    artifact = _build_artifact(dataset, best_pipeline, best_row, X_train)
    return EvaluationOutput(results, best_row, best_pipeline, artifact)


def _evaluate_classification(dataset: PreparedDataset, cv_splits: int) -> EvaluationOutput:
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset.y,
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    rows: list[dict[str, Any]] = []
    fitted_pipelines: dict[str, Pipeline] = {}

    for group, name, estimator in classification_models():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )

        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score = _positive_class_score(pipeline, X_test)

        row = {
            "model_group": group,
            "model_name": name,
            "cv_accuracy": float(np.mean(cv_scores["test_accuracy"])),
            "cv_precision": float(np.mean(cv_scores["test_precision"])),
            "cv_recall": float(np.mean(cv_scores["test_recall"])),
            "cv_f1": float(np.mean(cv_scores["test_f1"])),
            "cv_roc_auc": float(np.mean(cv_scores["test_roc_auc"])),
            "holdout_accuracy": float(accuracy_score(y_test, y_pred)),
            "holdout_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "holdout_recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "holdout_f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "holdout_roc_auc": float(roc_auc_score(y_test, y_score)),
        }
        rows.append(row)
        fitted_pipelines[name] = pipeline

    results = pd.DataFrame(rows).sort_values("holdout_f1", ascending=False)
    best_row = results.iloc[0].to_dict()
    best_pipeline = fitted_pipelines[best_row["model_name"]]

    y_pred_best = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    pd.DataFrame(
        cm,
        index=["actual_no", "actual_yes"],
        columns=["predicted_no", "predicted_yes"],
    ).to_csv(TABLES_DIR / "telco_churn_confusion_matrix.csv")

    artifact = _build_artifact(dataset, best_pipeline, best_row, X_train)
    artifact["metadata"]["labels"] = {0: "No", 1: "Yes"}
    return EvaluationOutput(results, best_row, best_pipeline, artifact)


def _positive_class_score(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(X)
    return pipeline.predict(X)


def _build_artifact(
    dataset: PreparedDataset,
    pipeline: Pipeline,
    best_row: dict[str, Any],
    X_train: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "pipeline": pipeline,
        "metadata": {
            "dataset_name": dataset.name,
            "task": dataset.task,
            "target": dataset.target,
            "model_name": best_row["model_name"],
            "metrics": best_row,
            "features": list(X_train.columns),
            "feature_schema": build_feature_schema(X_train),
        },
    }


def save_artifact(artifact: dict[str, Any], path) -> None:
    joblib.dump(artifact, path)
