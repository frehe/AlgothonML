import csv
import numpy as np
import pandas as pd
import yfinance as yf


def read_data() -> (pd.DataFrame, pd.DataFrame):
    df = pd.read_csv("data/prediction_train_with_targets.csv", sep=",", header=0)
    X_train = df.loc[:, df.columns != 'target_20']
    y_train = df.loc[:, df.columns == 'target_20']['target_20'].ravel()
    return X_train, y_train

def load_yfinance_data():
    return 


X_train, y_train = read_data()
pass
