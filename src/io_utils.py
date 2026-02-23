from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
from sklearn.pipeline import Pipeline


def save_artifacts(
    pipeline: Pipeline,
    feature_columns: List[str],
    target_column: str,
    model_name: str,
    metrics: Dict[str, float],
    out_dir: str = "models",
) -> Tuple[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_path = out_path / "property_price_model.joblib"
    meta_path = out_path / "property_price_model_meta.json"

    joblib.dump(pipeline, model_path)
    metadata = {
        "feature_columns": feature_columns,
        "target_column": target_column,
        "model_name": model_name,
        "metrics": metrics,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return str(model_path), str(meta_path)


def load_artifacts(
    model_path: str = "models/property_price_model.joblib",
    meta_path: str = "models/property_price_model_meta.json",
):
    pipeline = joblib.load(model_path)
    metadata = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return pipeline, metadata
