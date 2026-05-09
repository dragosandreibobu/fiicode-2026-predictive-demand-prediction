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
SAFE_LAGS = (7, 14, 21, 28)


@dataclass(frozen=True)
class RecursiveStatExperiment:
    name: str
    mode: str
    recent_window: int | None = None
    recent_weight: float = 0.0
    clip_quantile: float | None = None


EXPERIMENTS = (
    RecursiveStatExperiment(name="recursive_lag7", mode="lag7"),
    RecursiveStatExperiment(name="recursive_safe_lag_mean", mode="safe_lag_mean"),
    RecursiveStatExperiment(name="recursive_safe_lag_weighted", mode="safe_lag_weighted"),
    RecursiveStatExperiment(
        name="recursive_lag7_recent14_blend",
        mode="lag7",
        recent_window=14,
        recent_weight=0.30,
    ),
    RecursiveStatExperiment(
        name="recursive_safe_lag_mean_recent14_blend",
        mode="safe_lag_mean",
        recent_window=14,
        recent_weight=0.30,
    ),
    RecursiveStatExperiment(
        name="recursive_safe_lag_weighted_recent14_blend",
        mode="safe_lag_weighted",
        recent_window=14,
        recent_weight=0.30,
    ),
    RecursiveStatExperiment(
        name="recursive_safe_lag_weighted_clipped_p995",
        mode="safe_lag_weighted",
        clip_quantile=0.995,
    ),
)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_prediction(
    source: pd.DataFrame,
    target: pd.DataFrame,
    keys_chain: tuple[tuple[str, ...], ...],
    global_mean: float,
) -> pd.Series:
    predictions = pd.Series(np.nan, index=target.index, dtype=float)

    for keys in keys_chain:
        table = (
            source.groupby(list(keys), dropna=False)["demand"]
            .mean()
            .rename("__prediction")
            .reset_index()
        )
        merged = target[list(keys)].merge(table, on=list(keys), how="left")
        predictions = predictions.fillna(merged["__prediction"])

    return predictions.fillna(global_mean)


def lag_prediction(history: pd.DataFrame, target: pd.DataFrame, lag: int) -> pd.Series:
    lag_date = target["date"].iloc[0] - pd.Timedelta(days=lag)
    lookup = (
        history[history["date"] == lag_date][["store_id", "product_id", "demand"]]
        .rename(columns={"demand": f"lag_{lag}"})
    )
    merged = target[["store_id", "product_id"]].merge(
        lookup,
        on=["store_id", "product_id"],
        how="left",
    )
    return merged[f"lag_{lag}"]


def safe_lag_frame(history: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {f"lag_{lag}": lag_prediction(history, target, lag) for lag in SAFE_LAGS},
        index=target.index,
    )


def base_prediction_from_lags(lag_df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "lag7":
        return lag_df["lag_7"]
    if mode == "safe_lag_mean":
        return lag_df.mean(axis=1, skipna=True)
    if mode == "safe_lag_weighted":
        weights = pd.Series(
            {"lag_7": 0.45, "lag_14": 0.25, "lag_21": 0.20, "lag_28": 0.10}
        )
        weighted_sum = lag_df.mul(weights, axis=1).sum(axis=1, skipna=True)
        available_weight = lag_df.notna().mul(weights, axis=1).sum(axis=1)
        return weighted_sum / available_weight.replace(0, np.nan)
    raise ValueError(f"Unknown mode: {mode}")


def recent_prediction(
    history: pd.DataFrame,
    target: pd.DataFrame,
    window_days: int,
    global_mean: float,
) -> pd.Series:
    min_date = target["date"].iloc[0] - pd.Timedelta(days=window_days)
    recent_source = history[history["date"] >= min_date].copy()
    if recent_source.empty:
        recent_source = history
    return mean_prediction(
        source=recent_source,
        target=target,
        keys_chain=(
            ("store_id", "product_id"),
            ("product_id",),
            ("category",),
        ),
        global_mean=global_mean,
    )


def recursive_forecast(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    experiment: RecursiveStatExperiment,
    fallback_source: pd.DataFrame,
) -> pd.DataFrame:
    history = history_df.copy()
    predictions: list[pd.DataFrame] = []
    global_mean = float(fallback_source["demand"].mean())
    fallback_keys = (
        ("store_id", "product_id"),
        ("product_id",),
        ("category",),
    )
    upper_clip = None
    if experiment.clip_quantile is not None:
        upper_clip = float(fallback_source["demand"].quantile(experiment.clip_quantile))

    for current_date in sorted(future_df["date"].unique()):
        current = future_df[future_df["date"] == current_date].copy()
        lag_df = safe_lag_frame(history, current)
        base_pred = base_prediction_from_lags(lag_df, experiment.mode)
        fallback_pred = mean_prediction(fallback_source, current, fallback_keys, global_mean)
        pred = base_pred.fillna(fallback_pred)

        if experiment.recent_window is not None and experiment.recent_weight > 0:
            recent_pred = recent_prediction(
                history=history,
                target=current,
                window_days=experiment.recent_window,
                global_mean=global_mean,
            )
            pred = (1.0 - experiment.recent_weight) * pred + experiment.recent_weight * recent_pred

        pred = pred.clip(lower=0)
        if upper_clip is not None:
            pred = pred.clip(upper=upper_clip)

        current_output = current.copy()
        current_output["prediction"] = pred.to_numpy()
        predictions.append(current_output)

        next_history = current.copy()
        next_history["demand"] = pred.to_numpy()
        history = pd.concat([history, next_history], axis=0, ignore_index=True)

    return pd.concat(predictions, axis=0, ignore_index=True)


def save_submission(
    exports_root: Path,
    experiment_name: str,
    row_ids: pd.Series,
    predictions: pd.Series,
) -> Path:
    experiment_dir = exports_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    submission_path = experiment_dir / "submission.csv"
    named_path = experiment_dir / f"submission_{experiment_name}.csv"
    submission = pd.DataFrame({"row_id": row_ids, "demand": predictions.clip(lower=0)})
    submission.to_csv(submission_path, index=False)
    submission.to_csv(named_path, index=False)
    print(f"Saved {submission_path}")
    print(f"Saved {named_path}")
    return submission_path


def run_experiments(
    train_path: str | None,
    test_path: str | None,
    output_dir: str | None,
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

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("Validation period:", va_raw["date"].min(), "to", va_raw["date"].max())

    test_row_col = detect_row_column(test)
    results: list[dict[str, float | str]] = []

    for experiment in EXPERIMENTS:
        valid_eval = recursive_forecast(
            history_df=tr_raw,
            future_df=va_raw,
            experiment=experiment,
            fallback_source=tr_raw,
        )
        validation_rmse = rmse(valid_eval["demand"], valid_eval["prediction"])
        print(f"{experiment.name} validation RMSE: {validation_rmse:.6f}")

        test_eval = recursive_forecast(
            history_df=train,
            future_df=test,
            experiment=experiment,
            fallback_source=train,
        )
        test_eval = test_eval.sort_values(test_row_col).reset_index(drop=True)
        submission_path = save_submission(
            exports_root=exports_root,
            experiment_name=experiment.name,
            row_ids=test_eval[test_row_col],
            predictions=test_eval["prediction"],
        )
        results.append(
            {
                "experiment": experiment.name,
                "validation_rmse": validation_rmse,
                "submission_path": str(submission_path),
            }
        )

    results_df = pd.DataFrame(results).sort_values("validation_rmse")
    results_path = exports_root / "recursive_stat_baseline_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved {results_path}")
    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path")
    parser.add_argument("--test-path")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
