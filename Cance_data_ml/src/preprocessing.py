import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns="Unnamed: 32")
    return df

def remove_outliers_zscore(X, y, threshold=3):
    Z = (X - X.mean()) / X.std()
    mask = (Z.abs() > threshold).any(axis=1)
    return X[~mask], y[~mask]

def cap_outliers_iqr(X):
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return X.clip(lower=lower, upper=upper, axis=1)
