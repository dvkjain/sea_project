import numpy as np

def load_data(path):
    import pandas as pd

    if path.endswith(".xlsx"):
        data = pd.read_excel(path, engine="openpyxl").dropna()

    elif path.endswith(".csv"):
        data = pd.read_csv(path).dropna()

    elif path.endswith(".json"):
        data = pd.read_json(path).dropna()

    elif path.endswith(".h5") or path.endswith(".hdf5"):
        data = pd.read_hdf(path).dropna()

    else:
        raise ValueError("File format not supported. Please use .xlsx, .csv, .json, or .h5/.hdf5 files.", err=True)

    if data.empty:
        raise ValueError("Dataset is empty", err=True)
    
    if data.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns (features + target)", err=True)
    
    X = data.drop(columns=[data.columns[-1]])
    y = data[data.columns[-1]]

    return X, y

def scale_data(X, y, scaling, target_scaling):

    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    scaler = None
    target_scaler = None

    if scaling == "minmax":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

    elif scaling == "standard":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    elif scaling == "log":
        if np.any(X < 0):
            raise ValueError("Log scaling requires all feature values to be non-negative.", err=True)
        
        X_scaled = np.log1p(X)

    else: # case where scaling is "none"
        X_scaled = X

    y = np.asarray(y)

    if target_scaling == "minmax":
        target_scaler = MinMaxScaler()
        y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    elif target_scaling == "standard":
        target_scaler = StandardScaler()
        y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    elif target_scaling == "log":

        if np.any(y < 0):
            raise ValueError("Log scaling requires all target values to be non-negative.")
        
        y_scaled = np.log1p(y)

    else:  # case where target_scaling is "none"
        y_scaled = y

    return X_scaled, y_scaled, scaler, target_scaler
