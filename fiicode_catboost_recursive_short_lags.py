from __future__ import annotations

from fiicode_catboost_recursive_safe_lags import (
    build_config_from_args,
    parse_args,
    run_recursive_pipeline,
)


SHORT_LAGS = "1,2,3,7,14,21,28"


def main() -> None:
    args = parse_args()
    if args.run_label == "catboost_recursive_safe_lags":
        args.run_label = "catboost_recursive_short_lags"
    if args.output_file == "submission_catboost_recursive_safe_lags.csv":
        args.output_file = "submission_catboost_recursive_short_lags.csv"
    if args.lags is None:
        args.lags = SHORT_LAGS

    config = build_config_from_args(args)
    run_recursive_pipeline(
        config=config,
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
