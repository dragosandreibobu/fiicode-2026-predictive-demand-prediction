from fiicode_catboost_lag_pipeline import main


if __name__ == "__main__":
    main(
        default_use_rolling_features=True,
        default_run_label="catboost_lags_rolling",
        default_output_file="submission_catboost_lags_rolling.csv",
    )
