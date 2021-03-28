import pandas as pd
import numpy as np

test = pd.read_csv("sample_test.csv")
val = pd.read_csv("sample_validation.csv")

test_preds = test["prediction"].to_numpy()
val_preds = val["prediction"].to_numpy()

greater_idx = (test_preds>0.05)
smaller_idx = (test_preds<0.05)
test_preds[greater_idx] = 1.0
test_preds[smaller_idx] = -1.0
test_preds[(~greater_idx) & (~smaller_idx)] = 0

greater_idx = (val_preds>0.05)
smaller_idx = (val_preds<0.05)
val_preds[greater_idx] = 1.0
val_preds[smaller_idx] = -1.0
val_preds[(~greater_idx) & (~smaller_idx)] = 0

test["prediction"] = test_preds
val["prediction"] = val_preds

test.to_csv("BuyHighSellLow_test_ml.csv")
val.to_csv("BuyHighSellLow_validation_ml.csv")