from .base_model import BaseModel
import numpy as np
import joblib

class RegressionModel(BaseModel):
    def __init__(self, path, epochs, batch_size, neurons_per_layer, optimizer, activation, learning_rate, scaling, target_scaling):
        super().__init__(path, epochs, batch_size, neurons_per_layer, optimizer, activation, learning_rate, scaling)
        self.target_scaling = target_scaling

    def scaling_data(self):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        super().scaling_data()

        self.target_scaler = None
        if self.target_scaling == "log":
            self.y_train_scaled = np.log1p(self.y)

        elif self.target_scaling == "minmax":
            self.target_scaler = MinMaxScaler()
            self.y_train_scaled = self.target_scaler.fit_transform(self.y.reshape(-1, 1)).flatten()

        elif self.target_scaling == "standard":
            self.target_scaler = StandardScaler()
            self.y_train_scaled = self.target_scaler.fit_transform(self.y.reshape(-1, 1)).flatten()

        else:
            self.y_train_scaled = self.y
            
    def build_model(self):
        from sklearn.neural_network import MLPRegressor

        self.check_params()
        batch_size = self.X.shape[0] if self.batch_size == "all" or self.optimizer == "lbfgs" else int(self.batch_size)
        self.model = MLPRegressor(
            hidden_layer_sizes=self.neurons_per_layer,
            activation=self.activation,
            solver=self.optimizer,
            learning_rate_init=self.learning_rate,
            max_iter=self.epochs,
            batch_size=batch_size
        )
        return self.model

    def train_model(self):
        self.model.fit(self.X_train_scaled, self.y_train_scaled)
        return self.model

    def save_model(self, filename):
        to_save = {'model': self.model}
        to_save['task'] = 'regression'
        to_save['scaling_type'] = self.scaling
        to_save['target_scaling_type'] = self.target_scaling

        if self.scaler is not None:
            to_save['X_scaler'] = self.scaler
        if self.target_scaler is not None:
            to_save['y_scaler'] = self.target_scaler


        joblib.dump(to_save, filename)

    def __str__(self):
        base =  super().__str__()

        extra = ""

        if self.target_scaling != "none":
            extra += f"Target scaling method: {self.target_scaling}"

        return base+extra
