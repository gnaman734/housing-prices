from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42


@dataclass
class TrainResult:
    pipeline: Pipeline
    metrics: Dict[str, float]
    feature_importance: pd.DataFrame
    feature_columns: List[str]


def _get_feature_groups(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    feature_df = df.drop(columns=[target_col])
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def _build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], model_name: str) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "linear_regression":
        num_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=num_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def _get_model(model_name: str):
    if model_name == "linear_regression":
        return LinearRegression()
    if model_name == "decision_tree":
        return DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=12)
    if model_name == "random_forest":
        return RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def _extract_feature_importance(pipeline: Pipeline, top_k: int = 20) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importance = np.abs(coef.ravel()) if np.ndim(coef) > 1 else np.abs(coef)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    return imp_df.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)


def train_model(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    test_size: float = 0.2,
) -> TrainResult:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    clean_df = df.dropna(subset=[target_col]).copy()
    if clean_df.shape[0] < 20:
        raise ValueError("Dataset is too small. Use at least 20 rows with non-null target values.")

    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col]

    numeric_cols, categorical_cols = _get_feature_groups(clean_df, target_col)
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols, model_name)
    regressor = _get_model(model_name)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", regressor)])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
    }

    importance_df = _extract_feature_importance(pipeline)

    return TrainResult(
        pipeline=pipeline,
        metrics=metrics,
        feature_importance=importance_df,
        feature_columns=X.columns.tolist(),
    )


def prepare_inference_data(inference_df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    data = inference_df.copy()
    missing = [col for col in feature_columns if col not in data.columns]
    for col in missing:
        data[col] = np.nan

    return data[feature_columns]
