from datetime import datetime, timedelta

import csv
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn import preprocessing


NUM_DAYS = 30
feature_list = [str(k) for k in range(NUM_DAYS)]


def read_data(df) -> (pd.DataFrame, pd.DataFrame):
    # Split data and labels
    X_train = df.loc[:, df.columns != "Return"]
    y_train_df = df.loc[:, df.columns == "Return"]
    y_train = y_train_df["Return"].ravel()
    # X_train, y_train = X_train.loc[:199], y_train[:200]
    N = y_train.size
    # Scrape additional data for all timestamps
    features = pd.DataFrame(data=0, index=np.arange(N), columns=feature_list)
    for index, row in X_train.iterrows():
        # For each data row, identify the market and fill feature columns with past vals
        price_data = load_yfinance_data(
            market=row["Market"], date=row["Trade_date"], time=row["Trade_time"]
        )
        features.loc[index, :] = price_data
    complete_rows = features["0"].notna()
    features = features[complete_rows]
    X_train, y_train = X_train[complete_rows], y_train[complete_rows]
    # save_to_csv(features, y_train_df[complete_rows])
    return features, y_train


def read_data_val(df) -> pd.DataFrame:
    # Split data and labels
    X_train = df.loc[:, df.columns != "Return"]
    N = X_train.shape[0]
    # Scrape additional data for all timestamps
    features = pd.DataFrame(data=0, index=np.arange(1), columns=feature_list)
    i = 0
    for index, row in X_train.iterrows():
        # For each data row, identify the market and fill feature columns with past vals
        price_data = load_yfinance_data(
            market=row["Market"], date=row["Trade_date"], time=row["Trade_time"]
        )
        features.loc[i, :] = price_data
        i += 1
    complete_rows = features["0"].notna()
    features[~complete_rows] = 0
    bond_rows = ("bond" in X_train["Market"])
    commodity_rows = ("commodity" in X_train["Market"])
    currency_rows = ("currency" in X_train["Market"])
    stock_rows = ("stock" in X_train["Market"])
    scaler = preprocessing.StandardScaler()
    bonds = features.loc[bond_rows, :].values
    features.loc[bond_rows, :] = scaler.fit(bonds)
    return features


def load_yfinance_data(market: str, date: str, time: str) -> pd.DataFrame:
    ticker_symbol = ""
    if "bond" in market:
        ticker_symbol = "VBMFX"
    elif "commodity" in market:
        ticker_symbol = "DBC"
    elif "currency" in market:
        ticker_symbol = "DX-Y.NYB"
    elif "stock" in market:
        ticker_symbol = "^GSPC"
    timestamp_end = datetime.strptime(date, "%Y-%m-%d")
    timestamp_start = timestamp_end - timedelta(days=2 * NUM_DAYS)
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(interval="1d", start=timestamp_start, end=timestamp_end)[
        "Close"
    ]
    # Select the most recent available data up to the feature size
    data_array = data[-NUM_DAYS:].to_numpy()
    if data_array.size < NUM_DAYS:
        return np.zeros((NUM_DAYS,)) * np.nan
    return data_array


def save_to_csv(X: pd.DataFrame, y: pd.DataFrame):
    X.to_csv("data/X_train.csv")
    y.to_csv("data/y_train.csv")


# X_train, y_train = read_data()
# pass
