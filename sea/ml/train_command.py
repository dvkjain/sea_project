import click
from .regression_model import RegressionModel
from .classification_model import ClassificationModel

@click.command()
@click.argument("path")
@click.option("--random_state", type=int, default=42, show_default=True, help="Random state for reproducibility.")
@click.option("--task", type=click.Choice(["regression", "classification"]), default="regression", show_default=True, help="Task type: regression or classification.")
@click.option("--auto", "n_trials", type=int, default=None, help="Enable auto-tuning and set the number of trials. e.g., --auto 50")
@click.option("--encode", is_flag=True, help="Label encoding for target column. Applicable for classification tasks. If set, the target column will be encoded to integers.")
@click.option("--scaling", type=click.Choice(["none", "minmax", "standard", "log"]), default="none", show_default=True, help="Feature scaling method. When chosen, it will scale ALL x columns")
@click.option("--target_scaling", type=click.Choice(["none", "minmax", "standard", "log"]), default="none", show_default=True, help="Target variable scaling method")
@click.option("--epochs", "-e", default=100, type=int, show_default=True)
@click.option("--batch_size", default="32", show_default=True, help="Batch size for training. Use 'all' to use the entire dataset as a batch.")
@click.option("--neurons_per_layer", "--npl", default="1", show_default=True, type=str, help="Comma-separated list of neurons in each layer (e.g. 10,20,10). Default is 1 (one layer with 1 neuron). NOTE: the output layer is automatically added to the model")
@click.option("--learning_rate", default=0.01, show_default=True, type=float)
@click.option("--optimizer", "-o", type=click.Choice(["adam", "sgd", "lbfgs"]), default="adam", show_default=True, help="Optimizer to use for training.")
@click.option("--activation", "-a", type=click.Choice(["relu", "tanh", "identity", "logistic"]), default="relu", show_default=True, help="Activation function")
@click.option("--save", "-s", help="Saves the model (and scalers/encoder, if existant) in the selected filename.")
def train(path, random_state, task, n_trials, encode, scaling, target_scaling, epochs, batch_size, neurons_per_layer, learning_rate, optimizer, activation, save):
    """Train a neural network model on a dataset."""
    
    try:
        if task == "regression":
            if encode:
                click.echo("Label encoding is not applicable for regression tasks. It will be ignored.")
            
            regr = RegressionModel(path, random_state, epochs, batch_size, neurons_per_layer, optimizer, activation, learning_rate, scaling, target_scaling, n_trials)
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
                click.echo("Warning: target scaling is not applicable for classification tasks. It will be ignored.")
            
            classif = ClassificationModel(path, random_state, epochs, batch_size, neurons_per_layer, optimizer, activation, learning_rate, scaling, encode, n_trials)
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
