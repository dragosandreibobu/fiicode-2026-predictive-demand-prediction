# Key Submissions Documentation

## 1. Recursive CatBoost Anchor (2.0970)
- **Path:** `exports/recursive_catboost/submission.csv`
- **Description:** This model uses recursive forecasting with safe-lag features (lags of 7, 14, 21, 28 days) and rolling means. It treats predictions as history for subsequent steps.

## 2. LightGBM Recursive Model
- **Path:** `exports/lgb_recursive_safe_lags/submission.csv`
- **Description:** A gradient-boosted recursive model using the same feature engineering pipeline as the CatBoost anchor, but trained via LightGBM. Served as a diversity booster in the final ensemble.

## 3. Final Super Ensemble (2.0704)
- **Path:** `exports/FINAL_SUPER_ENSEMBLE.csv`
- **Description:** The ultimate ensemble blend created from a 60/20/20 weighted average of the Recursive CatBoost, the stabilized anchor model, and the Recursive XGBoost model. This was our highest-performing submission.
