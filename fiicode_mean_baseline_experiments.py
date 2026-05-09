from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from fiicode_catboost_lag_pipeline import (
    add_calendar_features,
    detect_row_column,
    find_competition_csvs,
)


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MeanExperiment:
    name: str
    output_file: str
    fallback_keys: tuple[tuple[str, ...], ...]


EXPERIMENTS = (
    MeanExperiment(
        name="product_store",
        output_file="submission_product_store_mean.csv",
        fallback_keys=(
            ("store_id", "product_id"),
            ("product_id",),
            ("category",),
        ),
    ),
    MeanExperiment(
        name="product_store_weekday",
        output_file="submission_product_store_weekday_mean.csv",
        fallback_keys=(
            ("store_id", "product_id", "dayofweek"),
            ("store_id", "product_id"),
            ("product_id", "dayofweek"),
            ("product_id",),
            ("category",),
        ),
    ),
    MeanExperiment(
        name="category_store_weekday",
        output_file="submission_category_store_weekday_mean.csv",
        fallback_keys=(
            ("store_id", "category", "dayofweek"),
            ("store_id", "category"),
            ("category", "dayofweek"),
            ("store_id", "product_id"),
            ("product_id",),
            ("category",),
        ),
    ),
    MeanExperiment(
        name="product_store_category_weekday_blend",
        output_file="submission_product_store_category_weekday_blend.csv",
        fallback_keys=(
            ("store_id", "product_id", "dayofweek"),
            ("store_id", "category", "dayofweek"),
            ("store_id", "product_id"),
            ("product_id", "dayofweek"),
            ("store_id", "category"),
            ("product_id",),
            ("category",),
        ),
    ),
)

RECENT_WINDOWS = (7, 14, 28, 56)
BLEND_WEIGHTS = (0.25, 0.50, 0.75)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_lookup(source: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    return (
        source.groupby(list(keys), dropna=False)["demand"]
        .mean()
        .rename("__prediction")
        .reset_index()
    )


def predict_with_fallbacks(
    source: pd.DataFrame,
    target: pd.DataFrame,
    fallback_keys: tuple[tuple[str, ...], ...],
    global_mean: float,
) -> pd.Series:
    predictions = pd.Series(np.nan, index=target.index, dtype=float)

    for keys in fallback_keys:
        table = build_lookup(source, keys)
        merged = target[list(keys)].merge(table, on=list(keys), how="left")
        predictions = predictions.fillna(merged["__prediction"])

    return predictions.fillna(global_mean).clip(lower=0)


def predict_recent_with_fallbacks(
    source: pd.DataFrame,
    target: pd.DataFrame,
    window_days: int,
    fallback_keys: tuple[tuple[str, ...], ...],
    global_mean: float,
) -> pd.Series:
    min_date = source["date"].max() - pd.Timedelta(days=window_days - 1)
    recent_source = source[source["date"] >= min_date].copy()
    if recent_source.empty:
        recent_source = source
    return predict_with_fallbacks(
        source=recent_source,
        target=target,
        fallback_keys=fallback_keys,
        global_mean=global_mean,
    )


def save_submission(
    exports_root: Path,
    experiment_name: str,
    output_file: str,
    row_ids: pd.Series,
    predictions: pd.Series,
) -> tuple[Path, Path]:
    experiment_dir = exports_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame({"row_id": row_ids, "demand": predictions.clip(lower=0)})
    submission_path = experiment_dir / "submission.csv"
    named_output_path = experiment_dir / output_file
    submission.to_csv(submission_path, index=False)
    submission.to_csv(named_output_path, index=False)
    print(f"Saved {submission_path}")
    print(f"Saved {named_output_path}")
    return submission_path, named_output_path


def run_experiments(
    train_path: str | None,
    test_path: str | None,
    output_dir: str | None,
    write_submission: bool,
) -> pd.DataFrame:
    exports_root = Path(output_dir) if output_dir else SCRIPT_DIR / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)

    resolved_train_path, resolved_test_path = find_competition_csvs(train_path, test_path)
    print("train_path:", resolved_train_path)
    print("test_path:", resolved_test_path)

    train = pd.read_csv(resolved_train_path)
    test = pd.read_csv(resolved_test_path)
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])

    origin_date = train["date"].min()
    train = add_calendar_features(train, origin_date)
    test = add_calendar_features(test, origin_date)

    last_train_date = train["date"].max()
    valid_start = last_train_date - pd.Timedelta(days=13)
    tr_raw = train[train["date"] < valid_start].copy()
    va_raw = train[train["date"] >= valid_start].copy()
    global_mean = float(tr_raw["demand"].mean())

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("Validation period:", va_raw["date"].min(), "to", va_raw["date"].max())

    results: list[dict[str, float | str]] = []
    valid_predictions: dict[str, pd.Series] = {}
    test_predictions: dict[str, pd.Series] = {}
    test_row_col = detect_row_column(test)

    for experiment in EXPERIMENTS:
        valid_pred = predict_with_fallbacks(
            source=tr_raw,
            target=va_raw,
            fallback_keys=experiment.fallback_keys,
            global_mean=global_mean,
        )
        valid_rmse = rmse(va_raw["demand"], valid_pred)
        print(f"{experiment.name} validation RMSE: {valid_rmse:.6f}")
        valid_predictions[experiment.name] = valid_pred

        full_pred = predict_with_fallbacks(
            source=train,
            target=test,
            fallback_keys=experiment.fallback_keys,
            global_mean=float(train["demand"].mean()),
        )
        test_predictions[experiment.name] = full_pred
        submission_path, named_output_path = save_submission(
            exports_root,
            experiment.name,
            experiment.output_file,
            test[test_row_col],
            full_pred,
        )

        results.append(
            {
                "experiment": experiment.name,
                "validation_rmse": valid_rmse,
                "submission_path": str(submission_path),
                "named_output_path": str(named_output_path),
            }
        )

    recent_fallbacks = (
        ("store_id", "product_id"),
        ("product_id",),
        ("category",),
    )
    recent_weekday_fallbacks = (
        ("store_id", "product_id", "dayofweek"),
        ("store_id", "product_id"),
        ("product_id", "dayofweek"),
        ("product_id",),
        ("category",),
    )

    for window_days in RECENT_WINDOWS:
        for name_suffix, fallback_keys in (
            ("recent", recent_fallbacks),
            ("recent_weekday", recent_weekday_fallbacks),
        ):
            name = f"product_store_{name_suffix}_{window_days}"
            output_file = f"submission_{name}_mean.csv"
            valid_pred = predict_recent_with_fallbacks(
                source=tr_raw,
                target=va_raw,
                window_days=window_days,
                fallback_keys=fallback_keys,
                global_mean=global_mean,
            )
            valid_rmse = rmse(va_raw["demand"], valid_pred)
            print(f"{name} validation RMSE: {valid_rmse:.6f}")
            valid_predictions[name] = valid_pred

            full_pred = predict_recent_with_fallbacks(
                source=train,
                target=test,
                window_days=window_days,
                fallback_keys=fallback_keys,
                global_mean=float(train["demand"].mean()),
            )
            test_predictions[name] = full_pred
            submission_path, named_output_path = save_submission(
                exports_root,
                name,
                output_file,
                test[test_row_col],
                full_pred,
            )
            results.append(
                {
                    "experiment": name,
                    "validation_rmse": valid_rmse,
                    "submission_path": str(submission_path),
                    "named_output_path": str(named_output_path),
                }
            )

    blend_pairs = (
        ("product_store", "product_store_recent_28"),
        ("product_store_weekday", "product_store_recent_weekday_56"),
        ("product_store_weekday", "product_store_recent_28"),
    )
    for left_name, right_name in blend_pairs:
        if left_name not in valid_predictions or right_name not in valid_predictions:
            continue
        for right_weight in BLEND_WEIGHTS:
            left_weight = 1.0 - right_weight
            name = (
                f"blend_{left_name}_{left_weight:.2f}_"
                f"{right_name}_{right_weight:.2f}"
            ).replace(".", "p")
            output_file = f"submission_{name}.csv"
            valid_pred = (
                left_weight * valid_predictions[left_name]
                + right_weight * valid_predictions[right_name]
            )
            valid_rmse = rmse(va_raw["demand"], valid_pred)
            print(f"{name} validation RMSE: {valid_rmse:.6f}")

            full_pred = (
                left_weight * test_predictions[left_name]
                + right_weight * test_predictions[right_name]
            )
            submission_path, named_output_path = save_submission(
                exports_root,
                name,
                output_file,
                test[test_row_col],
                full_pred,
            )
            results.append(
                {
                    "experiment": name,
                    "validation_rmse": valid_rmse,
                    "submission_path": str(submission_path),
                    "named_output_path": str(named_output_path),
                }
            )

    results_df = pd.DataFrame(results).sort_values("validation_rmse")
    results_path = exports_root / "mean_baseline_experiment_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved {results_path}")

    if write_submission:
        best_submission_path = Path(str(results_df.iloc[0]["submission_path"]))
        best_submission = pd.read_csv(best_submission_path)
        shared_submission_path = exports_root / "submission.csv"
        best_submission.to_csv(shared_submission_path, index=False)
        print(f"Saved {shared_submission_path}")
    else:
        print("Skipped writing submission.csv. Use --write-submission to enable it.")

    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path")
    parser.add_argument("--test-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--write-submission", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        write_submission=args.write_submission,
    )


if __name__ == "__main__":
    main()
