"""Train and evaluate the SF rent decision tree on disjoint city blocks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

TARGET = "monthly_rent_midpoint"
NUMERIC = [
    "bedroom_numeric", "bathroom_numeric", "square_footage_midpoint",
    "property_age", "unit_count", "submission_year", "latitude", "longitude",
]
CATEGORICAL = [
    "analysis_neighborhood", "supervisor_district", "case_type_name", "occupancy_type",
    "base_rent_includes_water_sewer", "base_rent_includes_natural_gas",
    "base_rent_includes_electricity", "base_rent_includes_refuse_recycling",
    "base_rent_includes_other_utilities",
]
FEATURES = NUMERIC + CATEGORICAL
AUDIT = [
    "unique_id", "block_num", "monthly_rent", "lower_bound_monthly_rent",
    "upper_bound_monthly_rent", TARGET,
]


def prepare_data(frame):
    """Retain positive, finite rent midpoints with valid bounds and block IDs."""
    frame = frame.copy()
    for name in NUMERIC + ["lower_bound_monthly_rent", "upper_bound_monthly_rent", TARGET]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = frame[TARGET].notna()
    positive = frame[TARGET].gt(0)
    bounds = (
        frame["lower_bound_monthly_rent"].notna()
        & frame["upper_bound_monthly_rent"].notna()
        & frame["lower_bound_monthly_rent"].ge(0)
        & frame["upper_bound_monthly_rent"].ge(frame["lower_bound_monthly_rent"])
    )
    groups = frame["block_num"].notna()
    exclusions = {
        "missing_or_open_ended_target": int((~finite).sum()),
        "nonpositive_target_with_known_midpoint": int((finite & ~positive).sum()),
        "invalid_or_missing_bounds": int((finite & positive & ~bounds).sum()),
        "missing_block_for_otherwise_valid_target": int((finite & positive & bounds & ~groups).sum()),
    }
    frame = frame.loc[finite & positive & bounds & groups].copy()
    if frame.empty:
        raise ValueError("No eligible positive rent-band midpoints remain.")
    midpoint = (frame["lower_bound_monthly_rent"] + frame["upper_bound_monthly_rent"]) / 2
    if not np.allclose(frame[TARGET], midpoint, rtol=0, atol=1e-10):
        raise ValueError("The target must equal the midpoint of its reported rent bounds.")
    for name in CATEGORICAL:
        values = frame[name].astype("string").str.strip().replace("", pd.NA).astype(object)
        frame[name] = values.where(values.notna(), np.nan)
    return frame, exclusions


def split_by_block(frame):
    """Hold out 20% of city blocks, including every row belonging to each block."""
    groups = frame["block_num"].astype("string").to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train, test = next(splitter.split(frame[FEATURES], frame[TARGET], groups))
    if not set(groups[train]).isdisjoint(groups[test]):
        raise ValueError("Training and testing city blocks overlap.")
    return train, test


def build_model():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL),
    ], sparse_threshold=0)
    return Pipeline([
        ("preprocess", preprocess),
        ("regressor", DecisionTreeRegressor(max_depth=12, min_samples_leaf=100, random_state=42)),
    ])


def calculate_metrics(actual, predicted):
    return {
        "mae_usd": float(mean_absolute_error(actual, predicted)),
        "rmse_usd": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(data_path, output_dir):
    source_hash = file_hash(data_path)
    raw = pd.read_csv(data_path, usecols=FEATURES + AUDIT, low_memory=False)
    source_rows = len(raw)
    frame, exclusions = prepare_data(raw)
    del raw
    train, test = split_by_block(frame)
    X = frame[FEATURES]
    y = frame[TARGET].to_numpy(dtype=float)
    groups = frame["block_num"].astype("string").to_numpy()
    rows = frame.index.to_numpy(dtype=np.int64)
    model = build_model()
    with threadpool_limits(limits=2):
        model.fit(X.iloc[train], y[train])
        predicted = model.predict(X.iloc[test])
        train_predicted = model.predict(X.iloc[train])
    baseline = DummyRegressor(strategy="median").fit(np.zeros((len(train), 1)), y[train])
    baseline_predicted = baseline.predict(np.zeros((len(test), 1)))
    heldout = calculate_metrics(y[test], predicted)
    baseline_metrics = calculate_metrics(y[test], baseline_predicted)
    manual_mae = float(np.abs(y[test] - predicted).mean())
    manual_r2 = float(1 - np.square(y[test] - predicted).sum() / np.square(y[test] - y[test].mean()).sum())
    checks = {
        "no_rent_derived_predictors": not bool(set(FEATURES) & set(AUDIT)),
        "target_matches_range_midpoint": True,
        "disjoint_city_blocks": set(groups[train]).isdisjoint(groups[test]),
        "disjoint_source_rows": set(rows[train]).isdisjoint(rows[test]),
        "manual_metrics_agree": bool(np.isclose(manual_mae, heldout["mae_usd"], rtol=0, atol=1e-10) and np.isclose(manual_r2, heldout["r2"], rtol=0, atol=1e-12)),
        "finite_predictions": bool(np.isfinite(predicted).all()),
        "source_unchanged": file_hash(data_path) == source_hash,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "decision_tree_rent_midpoint.joblib"
    joblib.dump(model, model_path, compress=3)
    checks["saved_model_predictions_identical"] = bool(np.array_equal(joblib.load(model_path).predict(X.iloc[test]), predicted))
    if not all(checks.values()):
        raise RuntimeError(f"Evaluation verification failed: {checks}")
    np.savez_compressed(output_dir / "holdout_predictions.npz", source_rows=rows[test], actual=y[test], predicted=predicted, baseline=baseline_predicted)
    np.savez_compressed(output_dir / "split_indices.npz", train_source_rows=rows[train], test_source_rows=rows[test])
    report = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": "Decision-tree rent-band midpoint evaluation with a fixed spatial holdout.",
        "source": data_path.name, "source_sha256": source_hash,
        "source_rows": source_rows, "eligible_rows": len(frame), "exclusions": exclusions,
        "target": "Monthly rent-band midpoint in USD, estimated from reported ranges rather than exact lease payments.",
        "features": FEATURES, "excluded_predictors": AUDIT,
        "split": {
            "method": "GroupShuffleSplit by city block (block_num)", "test_size_in_groups": 0.2, "random_state": 42,
            "train_rows": len(train), "test_rows": len(test),
            "train_blocks": len(set(groups[train])), "test_blocks": len(set(groups[test])), "overlapping_blocks": len(set(groups[train]) & set(groups[test])),
            "train_submission_years": sorted(map(int, frame["submission_year"].iloc[train].dropna().unique())),
            "test_submission_years": sorted(map(int, frame["submission_year"].iloc[test].dropna().unique())),
        },
        "model": {"class": "DecisionTreeRegressor", "max_depth": 12, "min_samples_leaf": 100, "random_state": 42, "criterion": "squared_error", "tuning": "None; configuration fixed before holdout evaluation."},
        "preprocessing": "Numeric median imputation and missing indicators; categorical Missing imputation and one-hot encoding. All fitted on training rows only.",
        "heldout_metrics": heldout, "training_metrics": calculate_metrics(y[train], train_predicted),
        "median_baseline_heldout_metrics": baseline_metrics,
        "mae_improvement_over_median_baseline_percent": 100 * (1 - heldout["mae_usd"] / baseline_metrics["mae_usd"]),
        "verification": checks,
        "limits": [
            "Targets are reported rent-band midpoints; missing/open-ended bands and zero-rent records are excluded.",
            "The holdout covers unseen city blocks within the observed 2022-2026 data; it does not measure future-year forecasting or current market-rent prediction.",
            "No test-set tuning was performed.",
        ],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Cleaned sf_rent_cleaned.csv exported by Data_Analysis.ipynb")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rent-evaluation"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
