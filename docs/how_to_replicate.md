# How to Replicate Results

This guide outlines how to reproduce our top-performing demand forecasting results.

## Prerequisites
- Install required packages:
  ```bash
  pip install -r requirements.txt
  ```

## Pipeline Execution
1. **Feature Engineering:** All recursive models share the same pipeline logic in `fiicode_catboost_lag_pipeline.py`.
2. **Recursive Training:** 
   - To replicate the 2.0970 anchor:
     ```bash
     python fiicode_catboost_recursive_safe_lags.py --iterations 1500
     ```
   - LightGBM and XGBoost variants follow the same command structure using their respective scripts in the root.
3. **Blending:**
   - To produce the final ensemble:
     ```bash
     python blend_models.py --cat exports/recursive_catboost/submission.csv --lgb exports/blend_cat_lgb_anchor_final.csv --xgb exports/xgb_recursive_safe_lags/submission.csv --out exports/FINAL_SUPER_ENSEMBLE.csv --weights 0.6,0.2,0.2
     ```
