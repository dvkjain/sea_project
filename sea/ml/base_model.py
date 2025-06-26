import pandas as pd
import os
import numpy as np

class BaseModel:
    def __init__(self, path, epochs, batch_size, neurons_per_layer, optimizer, activation, learning_rate, scaling=None):
        self.path = path
        self.epochs = epochs
        self.batch_size = batch_size
        self.neurons_per_layer = neurons_per_layer
        self.optimizer = optimizer
        self.activation = activation
        self.learning_rate = learning_rate
        self.scaling = scaling
        self.model = None
        self.scaler = None

    def load_data(self):
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"File not found: {self.path}")
        if self.path.endswith(".xlsx"):
            data = pd.read_excel(self.path, engine="openpyxl").dropna()
        elif self.path.endswith(".csv"):
            data = pd.read_csv(self.path).dropna()
        elif self.path.endswith(".json"):
            data = pd.read_json(self.path).dropna()
        elif self.path.endswith(".h5") or self.path.endswith(".hdf5"):
            data = pd.read_hdf(self.path).dropna()
        else:
            raise ValueError("File format not supported. Please use .xlsx, .csv, .json, or .h5/.hdf5 files.")
        if data.empty:
            raise ValueError("Dataset is empty")
        elif data.shape[1] < 2:
            raise ValueError(f"Dataset must have at least 2 columns (features + target). Found {data.shape[1]}")
        self.X = data.drop(columns=[data.columns[-1]])
        self.y = data[data.columns[-1]]
        return self.X, self.y

    def check_params(self):
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if self.batch_size != "all":
            try:
                if int(self.batch_size) <= 0:
                    raise ValueError("Batch size must be positive")
            except ValueError:
                raise ValueError("Batch size must be a positive integer or 'all'")
        if self.optimizer not in ["adam", "sgd", "lbfgs"]:
            raise ValueError(f"Invalid optimizer: {self.optimizer}. Choose from 'adam', 'sgd', or 'lbfgs'.")
        if isinstance(self.neurons_per_layer, str):
            try:
                self.neurons_per_layer = list(map(int, self.neurons_per_layer.split(",")))
            except ValueError:
                raise ValueError("Error: Invalid format for neurons_per_layer. Expected a comma-separated list of integers, e.g. '10,20,10'.")
        elif not isinstance(self.neurons_per_layer, list):
            raise ValueError("neurons_per_layer must be a string or a list of integers.")

    def scaling_data(self):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        if self.scaling == "minmax":
            self.scaler = MinMaxScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X)
            
        elif self.scaling == "standard":
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X)

        elif self.scaling == "log":
            if (self.X < 0).any().any():
                raise ValueError("Log scaling requires all feature values to be non-negative.")
            
            self.X_train_scaled = np.log1p(self.X)
        else:
            self.X_train_scaled = self.X

    def __str__(self):

        stats = (f"\nModel trained successfully on {self.path} with {len(self.X)} samples.\n"
            f"Epochs: {self.epochs}\n"
            f"Batch size: {self.batch_size}\n"
            f"Optimizer: {self.optimizer}\n"
            f"Activation function: {self.activation}\n"
            f"Learning rate: {self.learning_rate}\n"
            f"Neurons per hidden layer: {self.neurons_per_layer}\n")

        if self.scaling != "none":
            stats += f"\nScaling method: {self.scaling}"

        return stats
    