import click
import numpy as np
from .prepare_data import load_data, scale_data

def build_model(neurons_per_layer, activation, learning_rate, epochs, batch_size):

    from sklearn.neural_network import MLPRegressor

    if epochs <= 0:
        raise ValueError("Epochs must be positive")
    
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    model = MLPRegressor(hidden_layer_sizes=neurons_per_layer, activation=activation,
                        solver='adam', learning_rate_init=learning_rate, max_iter=epochs, batch_size=batch_size)
    
    return model

def train_model(model, X_train_scaled, y_train_scaled, target_scaling, target_scaler, scaling, scaler):

    model.fit(X_train_scaled, y_train_scaled)

    if target_scaling == "log":
        y_train_unscaled = np.expm1(y_train_scaled)
    elif target_scaler is not None:
        y_train_unscaled = target_scaler.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()
    else:
        y_train_unscaled = y_train_scaled

    if scaling == "log":
        X_train_unscaled = np.expm1(X_train_scaled)
    elif scaler is not None:
        X_train_unscaled = scaler.inverse_transform(X_train_scaled)
    else:
        X_train_unscaled = X_train_scaled

    return y_train_unscaled, X_train_unscaled

def save_model(model, filename, scaler, target_scaler, scaling, target_scaling, y_train_unscaled):

    import joblib

    to_save = {'model': model}

    if scaler is not None:
            to_save['X_scaler'] = scaler

    if target_scaler is not None and target_scaling != "log":
            to_save['y_scaler'] = target_scaler

    to_save['scaling_type'] = scaling
    to_save['target_scaling_type'] = target_scaling

    to_save['y_train_unscaled'] = y_train_unscaled

    joblib.dump(to_save, filename)
    
@click.command()
@click.argument("path")
@click.option("--scaling", type=click.Choice(["none", "minmax", "standard", "log"]), default="none", show_default=True, help="Feature scaling method. When chosen, it will scale ALL x columns")
@click.option("--target_scaling", type=click.Choice(["none", "minmax", "standard", "log"]), default="none", show_default=True, help="Target variable scaling method")
@click.option("--epochs", "-e", default=100, type=int, show_default=True)
@click.option("--batch_size", default=32, type=int, show_default=True)
@click.option("--neurons_per_layer", "--npl", default="1", show_default=True, type=str, help="Comma-separated list of neurons in each layer (e.g. 10,20,10). Default is 1 (one layer with 1 neuron). NOTE: the output layer is automatically added to the model")
@click.option("--learning_rate", default=0.01, show_default=True, type=float)
@click.option("--activation", "-a", type=click.Choice(["relu", "tanh", "identity", "logistic"]), default="relu", show_default=True, help="Activation function")
@click.option("--save", "-s", help="Saves the model (and scalers, if existant) in the selected filename.")
def train(path, scaling, target_scaling, epochs, batch_size, neurons_per_layer, learning_rate, activation, save):
    """Train a neural network model on a dataset."""

    try:
        X_train, y_train = load_data(path)
        X_train_scaled, y_train_scaled, scaler, target_scaler = scale_data(X_train, y_train, scaling, target_scaling)

        try:
            neurons_per_layer = list(map(int, neurons_per_layer.split(",")))
        except ValueError:
            raise click.ClickException(f"Error: Invalid format for neurons_per_layer. Expected a comma-separated list of integers, e.g. '10,20,10'.")
        
        model = build_model(neurons_per_layer, activation, learning_rate, epochs, batch_size)
        y_train_unscaled, X_train_unscaled = train_model(model, X_train_scaled, y_train_scaled, target_scaling, target_scaler, scaling, scaler)

        click.echo(f"\nModel trained successfully on {path} with {len(X_train)} samples.")
        click.echo(f"Epochs: {model.max_iter}")
        click.echo(f"Batch size: {model.batch_size}")
        click.echo(f"Activation function: {model.activation}")
        click.echo(f"Learning rate used: {model.learning_rate_init}")
        click.echo(f"Neurons per hidden layer: {model.hidden_layer_sizes}\n")

        if scaling != "none":
            click.echo(f"Scaling method: {scaling}")

        if target_scaling != "none":
            click.echo(f"Target scaling method: {target_scaling}")

        if save:
            save_model(model, save, scaler, target_scaler, scaling, target_scaling, y_train_unscaled)
            click.echo(f"Model and supported scalers saved as {save}.\n")

    except FileNotFoundError as e:
        raise click.ClickException(f"File not found: {e}")
    
    except ValueError as e:
        raise click.ClickException(f"Value error: {e}")
