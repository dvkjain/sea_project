import pandas as pd
import os

def load_data(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    if path.endswith(".xlsx"):
        data = pd.read_excel(path, engine="openpyxl").dropna()
    elif path.endswith(".csv"):
        data = pd.read_csv(path).dropna()
    elif path.endswith(".json"):
        data = pd.read_json(path).dropna()
    elif path.endswith(".h5") or path.endswith(".hdf5"):
        data = pd.read_hdf(path).dropna()
    else:
        raise ValueError("File format not supported. Please use .xlsx, .csv, .json, or .h5/.hdf5 files.")
    if data.empty:
        raise ValueError("Dataset is empty")
    elif data.shape[1] < 2:
        raise ValueError(f"Dataset must have at least 2 columns (features + target). Found {data.shape[1]}")
    X = data.drop(columns=[data.columns[-1]])
    y = data[data.columns[-1]]
    return X, y