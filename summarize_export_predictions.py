from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def summarize_submission(path: Path) -> dict[str, object]:
    submission = pd.read_csv(path)
    demand = submission["demand"]
    return {
        "experiment": path.parent.name,
        "path": str(path),
        "rows": len(submission),
        "mean": float(demand.mean()),
        "std": float(demand.std()),
        "min": float(demand.min()),
        "p01": float(demand.quantile(0.01)),
        "p50": float(demand.quantile(0.50)),
        "p99": float(demand.quantile(0.99)),
        "max": float(demand.max()),
        "zeros": int((demand == 0).sum()),
    }


def run_summary(exports_dir: str | None) -> pd.DataFrame:
    exports_root = Path(exports_dir) if exports_dir else SCRIPT_DIR / "exports"
    submission_paths = sorted(exports_root.glob("*/submission.csv"))
    summary = pd.DataFrame([summarize_submission(path) for path in submission_paths])

    if not summary.empty:
        summary = summary.sort_values(["mean", "experiment"]).reset_index(drop=True)

    output_path = exports_root / "submission_prediction_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Summarized {len(summary)} submissions")
    print(f"Saved {output_path}")
    if not summary.empty:
        print(summary[["experiment", "mean", "std", "min", "p50", "p99", "max", "zeros"]].head(10).to_string(index=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_summary(exports_dir=args.exports_dir)


if __name__ == "__main__":
    main()
