from .base_model import BaseModel
import joblib

class ClassificationModel(BaseModel):
    def __init__(self, path, epochs, batch_size, neurons_per_layer, optimizer, activation, learning_rate, scaling=None, encode=False):
        super().__init__(path, epochs, batch_size, neurons_per_layer, optimizer, activation, learning_rate, scaling)
        self.encode = encode
        self.target_encoder = None
        self.y_train_encoded = None

    def build_model(self):
        from sklearn.neural_network import MLPClassifier

        self.check_params()
        batch_size = self.X.shape[0] if self.batch_size == "all" or self.optimizer == "lbfgs" else int(self.batch_size)
        self.model = MLPClassifier(
            hidden_layer_sizes=self.neurons_per_layer,
            activation=self.activation,
            solver=self.optimizer,
            learning_rate_init=self.learning_rate,
            max_iter=self.epochs,
            batch_size=batch_size
        )
        return self.model
    
    def encode_target(self):
        from sklearn.preprocessing import LabelEncoder

        if self.encode:
            self.target_encoder = LabelEncoder()
            self.y_train_encoded = self.target_encoder.fit_transform(self.y)
        else:
            self.y_train_encoded = self.y
        return self.y_train_encoded

    def train_model(self):
        self.model.fit(self.X_train_scaled, self.y_train_encoded)
        return self.model

    def save_model(self, filename):
        to_save = {'model': self.model}
        to_save['task'] = 'classification'
        if self.scaler is not None:
            to_save['X_scaler'] = self.scaler
        to_save['scaling_type'] = self.scaling
        if self.target_encoder is not None:
            to_save['target_encoder'] = self.target_encoder
            to_save['y_train_encoded'] = self.y_train_encoded
        joblib.dump(to_save, filename)

    def __str__(self):
        return super().__str__()
    