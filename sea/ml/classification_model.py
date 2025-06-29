from .base_model import BaseModel
import joblib

class ClassificationModel(BaseModel):
    def __init__(self, path, random_state, epochs, batch_size, neurons_per_layer, optimizer, activation, learning_rate, scaling, encode, n_trials):
        super().__init__(path, random_state, epochs, batch_size, neurons_per_layer, optimizer, activation, learning_rate, scaling)
        self.random_state = random_state
        self.encode = encode
        self.target_encoder = None
        self.y_train_encoded = None
        self.tuning_trials = n_trials

    def build_model(self):
        from sklearn.neural_network import MLPClassifier

        self.check_params()
        batch_size = self.X_train.shape[0] if self.batch_size == "all" or self.optimizer == "lbfgs" else int(self.batch_size)
        self.model = MLPClassifier(
            hidden_layer_sizes=self.neurons_per_layer,
            activation=self.activation,
            solver=self.optimizer,
            learning_rate_init=self.learning_rate,
            max_iter=self.epochs,
            batch_size=batch_size,
            random_state=self.random_state)
    
    def encode_target(self):
        from sklearn.preprocessing import LabelEncoder

        if self.encode:
            self.target_encoder = LabelEncoder()
            self.y_train_encoded = self.target_encoder.fit_transform(self.y_train)

        else:
            self.y_train_encoded = self.y_train

    def hyperparameter_tuning(self):
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import uniform
        from sklearn.neural_network import MLPClassifier
        from halo import Halo

        spinner = Halo(text=f'Searching for the best model ({self.tuning_trials} trials)...', spinner='dots')

        param_distributions = {
            'solver': ['adam', 'sgd', 'lbfgs'],
            'batch_size': [16, 32, 64, self.X_train.shape[0]],
            'learning_rate_init': uniform(loc=0.0001, scale=0.0099),
            'activation': ['relu', 'tanh'],
            'hidden_layer_sizes': [(2), (4), (2, 4), (2, 2), (2, 4, 2)]
        }

        regressor = MLPClassifier(max_iter=self.epochs, random_state=self.random_state)

        tuner = RandomizedSearchCV(
            estimator=regressor,
            param_distributions=param_distributions,
            n_iter=self.tuning_trials,
            n_jobs=-1,
            cv=5,

            random_state=self.random_state,)

        try:
            with spinner:
                tuner.fit(self.X_train_scaled, self.y_train)
            
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
        self.model.fit(self.X_train_scaled, self.y_train_encoded)

    def save_model(self, filename):
        to_save = {'model': self.model}
        to_save['task'] = 'classification'
        to_save['scaling_type'] = self.scaling

        if self.scaler is not None:
            to_save['X_scaler'] = self.scaler

        if self.target_encoder is not None:
            to_save['target_encoder'] = self.target_encoder
            to_save['y_train_encoded'] = self.y_train_encoded

        joblib.dump(to_save, filename)

    def __str__(self):
        base =  super().__str__()
    
        if self.encode:
            extra = f"Label encoding was applied to target variable\n"

        else:
            extra = ""
        
        return base+extra
    