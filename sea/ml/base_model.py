import numpy as np
from .data_utils import load_data
import warnings

class BaseModel:
    def __init__(self, path, random_state, epochs, batch_size, neurons_per_layer, optimizer, activation, learning_rate, scaling):
        self.path = path
        self.random_state = random_state
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
        
        try:
            if isinstance(self.neurons_per_layer, str):
                self.neurons_per_layer = tuple(int(x) for x in self.neurons_per_layer.split(','))

        except ValueError:
            raise ValueError("Error: Invalid format for neurons_per_layer. Expected comma-separated integers, e.g. '10,20,10'.")

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
        n_samples = self.X_train.shape[0]

        if isinstance(self.batch_size, str) and self.batch_size.lower().startswith("all"):
            batch_info = f"all ({n_samples})"
        else:
            batch_info = str(self.batch_size)

        if isinstance(self.neurons_per_layer, (list, tuple)):
            neurons = ",".join(str(n) for n in self.neurons_per_layer)
        else:
            neurons = str(self.neurons_per_layer)

        stats = (
            f"\nModel trained successfully on {self.path} with {n_samples} samples.\n"
            f"Random state: {self.random_state}\n"
            f"Epochs: {self.epochs}\n"
            f"Batch size: {batch_info}\n"
            f"Optimizer: {self.optimizer}\n"
            f"Activation function: {self.activation}\n"
            f"Learning rate: {self.learning_rate}\n"
            f"Neurons per hidden layer: {neurons}\n"
        )

        if getattr(self, 'scaling', None) and self.scaling != "none":
            stats += f"\nScaling method: {self.scaling}\n"

        return stats
    