from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().copy()
    binary_columns = ["mainroad", "guestroom", "basement", "airconditioning", "prefarea"]
    for col in binary_columns:
        df[col] = df[col].map({"yes": 1, "no": 0})
    df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)
    return df


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run notebook baseline training pipeline")
    parser.add_argument("--data", required=True, help="Path to Housing.csv")
    parser.add_argument("--outdir", default="results", help="Directory to save metrics/model")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = preprocess(df)

    features = [
        "area",
        "bedrooms",
        "guestroom",
        "bathrooms",
        "mainroad",
        "prefarea",
        "stories",
        "parking",
        "basement",
        "airconditioning",
        "furnishingstatus_semi-furnished",
        "furnishingstatus_unfurnished",
    ]

    X = df[features]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics[name] = evaluate(model, X_test, y_test)

    best_name = min(metrics, key=lambda k: metrics[k]["RMSE"])
    best_model = models[best_name]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "notebook_baseline_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "best_model": best_name}, f, indent=2)

    joblib.dump(best_model, outdir / "best_house_price_model.pkl")

    print(json.dumps({"best_model": best_name, "metrics": metrics[best_name]}, indent=2))


if __name__ == "__main__":
    main()
