from .base_model import BaseModel
import numpy as np
import joblib

class RegressionModel(BaseModel):
    def __init__(self, path, random_state, epochs, batch_size, neurons_per_layer, optimizer, activation, learning_rate, scaling, target_scaling, n_trials):
        super().__init__(path, random_state, epochs, batch_size, neurons_per_layer, optimizer, activation, learning_rate, scaling)
        self.target_scaling = target_scaling
        self.tuning_trials = n_trials

    def scaling_data(self):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        super().scaling_data()

        self.target_scaler = None
        if self.target_scaling == "log":
            self.y_train_scaled = np.log1p(self.y_train)

        elif self.target_scaling == "minmax":
            self.target_scaler = MinMaxScaler()
            self.y_train_scaled = self.target_scaler.fit_transform(self.y_train.values.reshape(-1, 1)).flatten()

        elif self.target_scaling == "standard":
            self.target_scaler = StandardScaler()
            self.y_train_scaled = self.target_scaler.fit_transform(self.y_train.values.reshape(-1, 1)).flatten()

        else:
            self.y_train_scaled = self.y_train
            
    def build_model(self):
        from sklearn.neural_network import MLPRegressor

        self.check_params()
        batch_size = self.X_train.shape[0] if self.batch_size == "all" or self.optimizer == "lbfgs" else int(self.batch_size)
        self.model = MLPRegressor(
            hidden_layer_sizes=self.neurons_per_layer,
            activation=self.activation,
            solver=self.optimizer,
            learning_rate_init=self.learning_rate,
            max_iter=self.epochs,
            batch_size=batch_size,
            random_state=self.random_state)
        
    def hyperparameter_tuning(self):
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import uniform
        from sklearn.neural_network import MLPRegressor
        from halo import Halo

        spinner = Halo(text=f'Searching for the best model ({self.tuning_trials} trials)...', spinner='dots')

        param_distributions = {
            'solver': ['adam', 'sgd', 'lbfgs'],
            'batch_size': [16, 32, 64, self.X_train.shape[0]],
            'learning_rate_init': uniform(loc=0.0001, scale=0.0099),
            'activation': ['relu', 'tanh'],
            'hidden_layer_sizes': [(2), (4), (2, 4), (2, 2), (2, 4, 2)]
        }

        regressor = MLPRegressor(max_iter=self.epochs, random_state=self.random_state)

        tuner = RandomizedSearchCV(
            estimator=regressor,
            param_distributions=param_distributions,
            n_iter=self.tuning_trials,
            n_jobs=-1,
            cv=5,

            random_state=self.random_state,)

        try:
            with spinner:
                tuner.fit(self.X_train_scaled, self.y_train_scaled)
            
            spinner.succeed("Search finished.")

        except Exception as e:
            spinner.fail("Search failed.")
            raise e

        self.model = tuner.best_estimator_

        self.activation = self.model.activation
        self.learning_rate = self.model.learning_rate_init
        self.neurons_per_layer = self.model.hidden_layer_sizes
        self.optimizer = self.model.solver

        if self.model.solver == 'lbfgs':
            self.batch_size = f"all ({self.X_train.shape[0]})"
        else:
            self.batch_size = self.model.batch_size

    def train_model(self):
        self.model.fit(self.X_train_scaled, self.y_train_scaled)

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
            extra += f"Target scaling method: {self.target_scaling}\n"

        return base+extra
