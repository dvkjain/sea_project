import click
import yaml
from .regression_model import RegressionModel
from .classification_model import ClassificationModel

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

@click.command()
@click.argument("path")
@click.option("--config", "-c", type=click.Path(), default=None, help="Path to config YAML.")
@click.option("--random_state", type=int, default=None, help="Random state for reproducibility.")
@click.option("--task", type=click.Choice(["regression", "classification"]), default=None, help="Task type: regression or classification.")
@click.option("--auto", "n_trials", type=int, default=None, help="Enable auto-tuning and set the number of trials. e.g., --auto 50")
@click.option("--encode", is_flag=True, help="Label encoding for target column. Only for classification.")
@click.option("--scaling", type=click.Choice(["none", "minmax", "standard", "log"]), default=None, help="Feature scaling method.")
@click.option("--target_scaling", type=click.Choice(["none", "minmax", "standard", "log"]), default=None, help="Target variable scaling method.")
@click.option("--epochs", "-e", type=int, default=None)
@click.option("--batch_size", default=None, help="Batch size for training. Use 'all' for full-batch.")
@click.option("--neurons_per_layer", "--npl", default=None, type=str, help="Comma-separated list of neurons in each layer.")
@click.option("--learning_rate", default=None, type=float)
@click.option("--optimizer", "-o", type=click.Choice(["adam", "sgd", "lbfgs"]), default=None, help="Optimizer.")
@click.option("--activation", "-a", type=click.Choice(["relu", "tanh", "identity", "logistic"]), default=None, help="Activation function.")
@click.option("--save", "-s", help="Save the model to file.")
def train(path, config, random_state, task, n_trials, encode, scaling, target_scaling, epochs, batch_size, neurons_per_layer, learning_rate, optimizer, activation, save):
    """Train a neural network model on a dataset."""

    try:
        if config:
            cfg = load_config(config)
        else:
            cfg = {}

        train_cfg = cfg.get('train', {})

        random_state = random_state if random_state is not None else train_cfg.get('random_state', 42) # Default to 42 if not specified
        task = task if task is not None else train_cfg.get('task', 'regression') # Default to 'regression' if not specified
        encode = encode if encode is not None else train_cfg.get('encode', False) # Default to False if not specified
        scaling = scaling if scaling is not None else train_cfg.get('scaling', 'none') # Default to 'none' if not specified
        target_scaling = target_scaling if target_scaling is not None else train_cfg.get('target_scaling', 'none') # Default to 'none' if not specified
        epochs = epochs if epochs is not None else train_cfg.get('epochs', 100) # Default to 100 if not specified
        batch_size = batch_size if batch_size is not None else train_cfg.get('batch_size', '32') # Default to 32 if not specified
        neurons_per_layer = neurons_per_layer if neurons_per_layer is not None else train_cfg.get('neurons_per_layer', '1') # Default to '1' if not specified
        learning_rate = learning_rate if learning_rate is not None else train_cfg.get('learning_rate', 0.01) # Default to 0.01 if not specified
        optimizer = optimizer if optimizer is not None else train_cfg.get('optimizer', 'adam') # Default to 'adam' if not specified
        activation = activation if activation is not None else train_cfg.get('activation', 'relu') # Default to 'relu' if not specified

        if task == "regression":
            if encode:
                click.echo("Label encoding ignored for regression.")

            regr = RegressionModel(
                path, random_state, epochs, batch_size,
                neurons_per_layer, optimizer, activation,
                learning_rate, scaling, target_scaling, n_trials
            )

            regr.check_params()
            regr.scaling_data()
            if n_trials is not None:
                regr.hyperparameter_tuning()
            else:
                regr.build_model()

            regr.train_model()
            click.echo(str(regr))

            if save:
                regr.save_model(save)
                click.echo(f"Model saved to {save}")

        elif task == "classification":
            if target_scaling != "none":
                click.echo("Warning: target scaling ignored for classification.")

            classif = ClassificationModel(
                path, random_state, epochs, batch_size,
                neurons_per_layer, optimizer, activation,
                learning_rate, scaling, encode, n_trials
            )

            classif.check_params()
            classif.scaling_data()
            classif.encode_target()

            if n_trials is not None:
                classif.hyperparameter_tuning()
            else:
                classif.build_model()

            classif.train_model()
            click.echo(str(classif))

            if save:
                classif.save_model(save)
                click.echo(f"Model saved to {save}")

    except FileNotFoundError as e:
        raise click.ClickException(f"File not found: {e}")
    except ValueError as e:
        raise click.ClickException(f"Value error: {e}")
