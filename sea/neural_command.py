import click
import numpy as np

def load_data(path, test_size, random_state):

    import pandas as pd
    from sklearn.model_selection import train_test_split

    if path.endswith(".xlsx"):

        data = pd.read_excel(path, engine="openpyxl").dropna()

        X = data.drop(columns=[data.columns[-1]])
        y = data[data.columns[-1]]

    elif path.endswith(".csv"):

        data = pd.read_csv(path).dropna()

        X = data.drop(columns=[data.columns[-1]])
        y = data[data.columns[-1]]

    else:
        raise ValueError("File format not supported. Please use .xlsx or .csv files.")

    if data.empty:
        raise ValueError("Dataset is empty")
    
    if data.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns (features + target)")
    
    if test_size <= 0 or test_size >= 1:
        raise ValueError("Test_size must be between 0 and 1")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def scale_data(X_train, y_train, X_test, y_test, scaling, target_scaling):

    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    scaler = None
    target_scaler = None

    if scaling == "minmax":
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    elif scaling == "standard":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    elif scaling == "log":
        if np.any(X_train < 0) or np.any(X_test < 0):
            raise ValueError("Log scaling requires all feature values to be non-negative.")
        
        X_train_scaled = np.log1p(X_train)
        X_test_scaled = np.log1p(X_test)
    
    else: # Handle the "none" case
        X_train_scaled = X_train
        X_test_scaled = X_test

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if target_scaling == "minmax":
        target_scaler = MinMaxScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    elif target_scaling == "standard":
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    elif target_scaling == "log":

        if np.any(y_train < 0) or np.any(y_test < 0):
            raise ValueError("Log scaling requires all target values to be non-negative.")
        
        y_train_scaled = np.log1p(y_train)
        y_test_scaled = np.log1p(y_test)

    else:  # Handle the "none" case
        y_train_scaled = y_train
        y_test_scaled = y_test

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler, target_scaler

def build_model(X_train, neurons_per_layer, activation, learning_rate, epochs, batch_size):

    from sklearn.neural_network import MLPRegressor

    if epochs <= 0:
        raise ValueError("Epochs must be positive")
    
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    model = MLPRegressor(hidden_layer_sizes=neurons_per_layer, activation=activation,
                         solver='adam', learning_rate_init=learning_rate, max_iter=epochs, batch_size=batch_size)
    
    return model

def train_and_evaluate_model(model, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, target_scaling, target_scaler):

    from sklearn.metrics import mean_absolute_error, r2_score

    model.fit(X_train_scaled, y_train_scaled)

    y_pred_scaled = model.predict(X_test_scaled)

    if target_scaling == "log":
        y_pred_unscaled = np.expm1(y_pred_scaled)
        y_test_unscaled = np.expm1(y_test_scaled)

    elif target_scaler is not None:
        y_pred_unscaled = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_unscaled = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    else:
        y_pred_unscaled = y_pred_scaled
        y_test_unscaled = y_test_scaled

    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    r2 = r2_score(y_test_unscaled, y_pred_unscaled)
    
    click.echo(f"""
               ------------RESULTS------------
               Test MAE: {mae}
               Test R²: {r2}
                """)

    return y_test_unscaled, y_pred_unscaled

def save_model(model, filename, scaler, target_scaler, scaling, target_scaling):

    import joblib

    to_save = {'model': model}
    if scaler is not None:
        to_save['X_scaler'] = scaler
    if target_scaler is not None and target_scaling != "log":
        to_save['y_scaler'] = target_scaler

    if scaling == "log" or target_scaling == "log":
        click.echo("Log scaler was used, but can't be saved. Only the model and supported scalers will be saved.")

    joblib.dump(to_save, filename)
    click.echo(f"Model and supported scalers saved as {filename}.")

@click.command()
@click.argument("path")
@click.option("--random_state", default=None, show_default=True, type=int, help="Set a seed for reproducibility. Default is None for random splits")
@click.option("--test_size", default=0.2, show_default=True, type=float, help="Set a size for the testing data.")
@click.option("--scaling", type=click.Choice(["none", "minmax", "standard", "log"]), default="none", show_default=True, help="Feature scaling method. When chosen, it will scale ALL x columns")
@click.option("--target_scaling", type=click.Choice(["none", "minmax", "standard", "log"]), default="none", show_default=True, help="Target variable scaling method")
@click.option("--epochs", "-e", default=100, type=int, show_default=True)
@click.option("--batch_size", default=32, type=int, show_default=True)
@click.option("--neurons_per_layer", "--npl", default="1", show_default=True, type=str, help="Comma-separated list of neurons in each layer (e.g. 10,20,10). Default is 1 (one layer with 1 neuron). NOTE: the output layer is automatically added to the model")
@click.option("--learning_rate", default=0.01, show_default=True, type=float)
@click.option("--activation", "-a", type=click.Choice(["relu", "tanh", "identity", "logistic"]), default="relu", show_default=True, help="Activation function")
@click.option("--save", "-s", help="Saves the model (and scalers, if existant) in the selected filename.")
def neural(path, scaling, target_scaling, random_state, test_size, epochs, batch_size, neurons_per_layer, learning_rate, activation, save):
    
    """Allows the user to train a neural network model on a dataset.\n 
    The dataset must be in .csv or .xlsx format.\n
    The target variable must be the last column."""

    try:
        X_train, X_test, y_train, y_test = load_data(path, test_size, random_state)
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler, target_scaler = scale_data(X_train, y_train, X_test, y_test, scaling, target_scaling)

        # Convert neurons_per_layer string into a list of integers
        neurons_per_layer = list(map(int, neurons_per_layer.split(",")))

        model = build_model(X_train, neurons_per_layer, activation, learning_rate, epochs, batch_size)
        y_test_unscaled, y_pred_unscaled = train_and_evaluate_model(model, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, target_scaling, target_scaler)
        
        click.echo(f"Epochs: {model.max_iter}")
        click.echo(f"Batch size: {model.batch_size}")
        click.echo(f"Activation function: {model.activation}")
        click.echo(f"Learning rate used: {model.learning_rate_init}")
        click.echo(f"Neurons per hidden layer: {model.hidden_layer_sizes}")

        if scaling != "none":
            click.echo(f"Scaling method: {scaling}")
            
        if target_scaling != "none":
            click.echo(f"Target scaling method: {target_scaling}")

    except FileNotFoundError as exc:
        click.echo(f"Error: File '{path}' not found.", err=True)
        raise SystemExit(1) from exc
    
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e
    
    save_model(model, save, scaler, target_scaler, scaling, target_scaling)

if __name__ == "__main__":
    neural()
