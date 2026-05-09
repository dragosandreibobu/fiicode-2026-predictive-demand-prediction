#!/usr/bin/env python
# coding: utf-8

# # FiiCode 2026 Final — Product-Store Average Baseline (FIXED VERSION)
# 
# This notebook builds a **simple forecasting baseline** for the Kaggle competition.
# 
# The method is:
# 
# > For each `(product_id, store_id)` pair, predict the average historical demand.
# 
# This is a **mean aggregation baseline**.

# ## 1. Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ## 2. Load data

# In[ ]:


# Detect file paths
train_path = None
test_path = None

for dirname, _, filenames in os.walk("/kaggle/input"):
    if "train_final.csv" in filenames:
        train_path = os.path.join(dirname, "train_final.csv")
    if "test_final.csv" in filenames:
        test_path = os.path.join(dirname, "test_final.csv")

if train_path is None or test_path is None:
    raise FileNotFoundError(
        f"Could not find train/test CSVs under /kaggle/input. "
        f"train_path={train_path}, test_path={test_path}"
    )

print(f"Loading data from {train_path}...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


# ## 3. Preprocessing

# In[ ]:


# Convert date column
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

print("Train date range:", train['date'].min(), "to", train['date'].max())
print("Test date range: ", test['date'].min(), "to", test['date'].max())


# ## 4. Create a last-14-days validation split

# In[ ]:


last_train_date = train["date"].max()
valid_start = last_train_date - pd.Timedelta(days=13)

tr = train[train["date"] < valid_start].copy()
va = train[train["date"] >= valid_start].copy()

print("Training period:", tr["date"].min(), "to", tr["date"].max())
print("Validation period:", va["date"].min(), "to", va["date"].max())

print("tr shape:", tr.shape)
print("va shape:", va.shape)


# ## 5. Build fallback means from the training split

# In[ ]:


product_store_mean = (
    tr.groupby(["store_id", "product_id"])["demand"]
      .mean()
      .rename("product_store_mean")
      .reset_index()
)

product_mean = (
    tr.groupby("product_id")["demand"]
      .mean()
      .rename("product_mean")
      .reset_index()
)

category_mean = (
    tr.groupby("category")["demand"]
      .mean()
      .rename("category_mean")
      .reset_index()
)

global_mean = tr["demand"].mean()

print("Global mean demand:", global_mean)


# ## 6. Predict validation set

# In[ ]:


va_pred = va.copy()

va_pred = va_pred.merge(product_store_mean, on=["store_id", "product_id"], how="left")
va_pred = va_pred.merge(product_mean, on="product_id", how="left")
va_pred = va_pred.merge(category_mean, on="category", how="left")

va_pred["prediction"] = va_pred["product_store_mean"]
va_pred["prediction"] = va_pred["prediction"].fillna(va_pred["product_mean"])
va_pred["prediction"] = va_pred["prediction"].fillna(va_pred["category_mean"])
va_pred["prediction"] = va_pred["prediction"].fillna(global_mean)

va_pred["prediction"] = va_pred["prediction"].clip(lower=0)

rmse = rmse_score(va_pred["demand"], va_pred["prediction"])
print("Validation RMSE:", rmse)


# ## 7. Train on full training data

# In[ ]:


full_product_store_mean = (
    train.groupby(["store_id", "product_id"])["demand"]
         .mean()
         .rename("product_store_mean")
         .reset_index()
)

full_product_mean = (
    train.groupby("product_id")["demand"]
         .mean()
         .rename("product_mean")
         .reset_index()
)

full_category_mean = (
    train.groupby("category")["demand"]
         .mean()
         .rename("category_mean")
         .reset_index()
)

full_global_mean = train["demand"].mean()

print("Full global mean demand:", full_global_mean)


# ## 8. Predict the test set

# In[ ]:


test_pred = test.copy()

test_pred = test_pred.merge(full_product_store_mean, on=["store_id", "product_id"], how="left")
test_pred = test_pred.merge(full_product_mean, on="product_id", how="left")
test_pred = test_pred.merge(full_category_mean, on="category", how="left")

test_pred["demand"] = test_pred["product_store_mean"]
test_pred["demand"] = test_pred["demand"].fillna(test_pred["product_mean"])
test_pred["demand"] = test_pred["demand"].fillna(test_pred["category_mean"])
test_pred["demand"] = test_pred["demand"].fillna(full_global_mean)

test_pred["demand"] = test_pred["demand"].clip(lower=0)

print("Test predictions complete.")


# ## 9. Build submission file

# In[ ]:


if "row_id" in test_pred.columns:
    row_col = "row_id"
elif "row_ID" in test_pred.columns:
    row_col = "row_ID"
else:
    raise ValueError("No row_id / row_ID column found.")

submission = pd.DataFrame({
    "row_id": test_pred[row_col],
    "demand": test_pred["demand"]
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")

