import click
from .prepare_data import load_data

def plot_predictions(y_test, y_pred_unscaled):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred_unscaled, alpha=0.5)
    sns.lineplot(x=y_test, y=y_test, color='red', linestyle='-', label='Perfect Prediction')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predictions vs Actual Values")
    plt.show()

def show_metrics(metrics, y_test, y_pred_unscaled):
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
        r2_score, median_absolute_error, explained_variance_score, max_error,
        mean_squared_log_error
    )

    metrics = metrics.lower()
    metric_functions = {
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "mape": mean_absolute_percentage_error,
        "r2": r2_score,
        "medae": median_absolute_error,
        "evs": explained_variance_score,
        "max_error": max_error,
        "msle": mean_squared_log_error,
        }

    for metric in metrics.split(","):
        metric = metric.strip().lower()
        if metric in metric_functions:
            value = metric_functions[metric](y_test, y_pred_unscaled)
            click.echo(f"{metric.upper()}: {value}")
            valid_metrics += 1
        else:
            click.echo(f"Warning: Metric '{metric}' is not recognized.")
    
    if valid_metrics == 0:
        raise click.ClickException("No valid metrics provided.")

@click.command()
@click.argument("model_path")
@click.argument("data_path")
@click.option("--metrics", help="Metric to evaluate the model.")
@click.option("--plot", is_flag=True, help="Plot the predictions vs actual values.")
def evaluate(model_path, data_path, metrics, plot):
    """Evaluate a pre-trained model on a new dataset."""

    import joblib
    import numpy as np

    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data.get('X_scaler')
        target_scaler = model_data.get('y_scaler')
        target_scaling = model_data.get('target_scaling_type', 'none')
        scaling = model_data.get('scaling_type', 'none')


        X_test, y_test = load_data(data_path)

        if scaling == "log":
            X_test_scaled = np.log1p(X_test)
        elif scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test

        y_pred_scaled = model.predict(X_test_scaled)

        if target_scaling == "log":
            y_pred_unscaled = np.expm1(y_pred_scaled)
        elif target_scaler is not None:
            y_pred_unscaled = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        else:
            y_pred_unscaled = y_pred_scaled

        click.echo(f"Scaling used: {model_data.get('scaling_type', 'none')}\nTarget scaling used: {model_data.get('target_scaling_type', 'none')}")

    except FileNotFoundError as exc:
        raise click.ClickException(f"File not found: {exc}") from exc
    
    except ValueError as e:
        raise click.ClickException(f"Value error: {e}") from e

    if metrics:
        show_metrics(metrics, y_test, y_pred_unscaled)

    if plot:
        plot_predictions(y_test, y_pred_unscaled)
if __name__ == "__main__":
    evaluate()
