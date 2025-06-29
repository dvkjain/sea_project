import numpy as np
from .data_utils import load_data
import warnings

class BaseModel:
    def __init__(self, path, randon_state, epochs, batch_size, neurons_per_layer, optimizer, activation, learning_rate, scaling):
        self.path = path
        self.random_state = randon_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.neurons_per_layer = neurons_per_layer
        self.optimizer = optimizer
        self.activation = activation
        self.learning_rate = learning_rate
        self.scaling = scaling
        self.model = None
        self.scaler = None

        self.X_train, self.y_train = load_data(self.path)

    def check_params(self):
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        
        if self.batch_size != "all":

            if int(self.batch_size) <= 0:
                raise ValueError("Batch size must be positive")
            
            elif self.optimizer == "lbfgs":
                warnings.warn("Since optimizer is 'lbfgs', batch size will be set to 'all'.")
                self.batch_size = "all"
        
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
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            
        elif self.scaling == "standard":
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)

        elif self.scaling == "log":
            if (self.X_train < 0).any().any():
                raise ValueError("Log scaling requires all feature values to be non-negative.")
            
            self.X_train_scaled = np.log1p(self.X_train)
        else:
            self.X_train_scaled = self.X_train

    def __str__(self):

        stats = (f"\nModel trained successfully on {self.path} with {len(self.X_train)} samples.\n"
            f"Random state: {self.random_state}\n"
            f"Epochs: {self.epochs}\n"
            f"Batch size: {self.batch_size if self.batch_size != "all" else f"Batch size: all ({self.X_train.shape[0]})"}\n"
            f"Optimizer: {self.optimizer}\n"
            f"Activation function: {self.activation}\n"
            f"Learning rate: {self.learning_rate}\n"
            f"Neurons per hidden layer: {self.neurons_per_layer}\n")

        if self.scaling != "none":
            stats += f"\nScaling method: {self.scaling}\n"

        return stats
    