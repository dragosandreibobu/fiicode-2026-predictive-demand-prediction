from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fiicode_catboost_lag_pipeline import detect_row_column, find_competition_csvs


SCRIPT_DIR = Path(__file__).resolve().parent
EXPECTED_COLUMNS = ["row_id", "demand"]


def validate_submission(path: Path, expected_row_ids: pd.Series) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path),
        "status": "ok",
        "rows": 0,
        "min_demand": None,
        "max_demand": None,
        "issue": "",
    }

    try:
        submission = pd.read_csv(path)
    except Exception as exc:
        result["status"] = "failed"
        result["issue"] = f"read error: {exc}"
        return result

    result["rows"] = len(submission)

    if list(submission.columns) != EXPECTED_COLUMNS:
        result["status"] = "failed"
        result["issue"] = f"bad columns: {list(submission.columns)}"
        return result

    if len(submission) != len(expected_row_ids):
        result["status"] = "failed"
        result["issue"] = f"bad row count: {len(submission)} expected {len(expected_row_ids)}"
        return result

    if not submission["row_id"].reset_index(drop=True).equals(expected_row_ids.reset_index(drop=True)):
        result["status"] = "failed"
        result["issue"] = "row_id order/content mismatch"
        return result

    if submission["demand"].isna().any():
        result["status"] = "failed"
        result["issue"] = "NaN demand values"
        return result

    result["min_demand"] = float(submission["demand"].min())
    result["max_demand"] = float(submission["demand"].max())

    if (submission["demand"] < 0).any():
        result["status"] = "failed"
        result["issue"] = "negative demand values"
        return result

    return result


def run_validation(exports_dir: str | None, test_path: str | None) -> pd.DataFrame:
    exports_root = Path(exports_dir) if exports_dir else SCRIPT_DIR / "exports"
    _, resolved_test_path = find_competition_csvs(test_path=test_path)
    test = pd.read_csv(resolved_test_path)
    row_col = detect_row_column(test)
    expected_row_ids = test[row_col]

    submission_paths = sorted(exports_root.glob("*/submission.csv"))
    results = [validate_submission(path, expected_row_ids) for path in submission_paths]
    results_df = pd.DataFrame(results)

    output_path = exports_root / "submission_validation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Validated {len(results_df)} submissions")
    print(results_df["status"].value_counts(dropna=False).to_string())
    print(f"Saved {output_path}")

    failed = results_df[results_df["status"] != "ok"]
    if not failed.empty:
        print(failed[["path", "issue"]].to_string(index=False))
        raise SystemExit(1)

    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports-dir")
    parser.add_argument("--test-path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_validation(exports_dir=args.exports_dir, test_path=args.test_path)


if __name__ == "__main__":
    main()
