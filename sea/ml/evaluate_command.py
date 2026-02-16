import click
import numpy as np
import joblib
import matplotlib.pyplot as plt
from .data_utils import load_data

def plot_predictions_classification(y_test_encoded, y_pred_encoded, class_names=None):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    labels = np.unique(np.concatenate([y_test_encoded, y_pred_encoded]))
    cm = confusion_matrix(y_test_encoded, y_pred_encoded, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names is not None else labels,
                yticklabels=class_names if class_names is not None else labels)
    
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def plot_predictions_regression(y_test, y_pred_unscaled):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred_unscaled, alpha=0.5,label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='-', label='Actual Values (Perfect Prediction)')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predictions vs Actual Values")
    plt.legend()
    plt.show()

def show_metrics_regression(metrics, y_test, y_pred_unscaled):
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
    
    valid_metrics = 0

    for metric in metrics.split(","):
        metric = metric.strip().lower()
        if metric in metric_functions:
            value = metric_functions[metric](y_test, y_pred_unscaled)
            click.echo(f"{metric.upper()}: {value}")
            valid_metrics += 1
        else:
            click.echo(f"Warning: Metric '{metric}' is not recognized for regression.")

    if valid_metrics == 0:
        raise click.ClickException("No valid metrics provided.")
    
    click.echo()

def show_metrics_classification(metrics, y_test_encoded, y_pred_encoded):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    metrics = metrics.lower()
    metric_functions = {
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0),
    "roc_auc": roc_auc_score,
    "confusion_matrix": confusion_matrix,
    "classification_report": classification_report
}

    valid_metrics = False

    for metric in metrics.split(","):
        metric = metric.strip().lower()
        if metric in metric_functions:
            value = metric_functions[metric](y_test_encoded, y_pred_encoded)
            click.echo(f"{metric.upper()}: {value}")
            valid_metrics = True
            
        else:
            click.echo(f"Warning: Metric '{metric}' is not recognized for classification.")

    if not valid_metrics:
        raise click.ClickException("No valid metrics provided.")
    click.echo()

@click.command()
@click.argument("model_path")
@click.argument("data_path")
@click.option("--metrics", help="Metric to evaluate the model.")
@click.option("--plot", is_flag=True, help="Plot the predictions vs actual values.")
def evaluate(model_path, data_path, metrics, plot):
    """Evaluate a pre-trained model on a new dataset."""

    model_data = joblib.load(model_path)
    model = model_data['model']
    task = model_data.get('task')

    X_test, y_test = load_data(data_path)

    scaler = model_data.get('X_scaler')
    scaling = model_data.get('scaling_type', 'none')
    
    if scaling == "log":

        X_test_scaled = np.log1p(X_test)

    elif scaler is not None:
        X_test_scaled = scaler.transform(X_test)

    else:
        X_test_scaled = X_test

    try:
        if task == "regression":

            target_scaler = model_data.get('y_scaler')
            target_scaling = model_data.get('target_scaling_type', 'none')

            y_pred_scaled = model.predict(X_test_scaled)

            if target_scaling == "log":
                y_pred_unscaled = np.expm1(y_pred_scaled)
            elif target_scaler is not None:
                y_pred_unscaled = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            else:
                y_pred_unscaled = y_pred_scaled

            click.echo(f"\nScaling used: {model_data.get('scaling_type', 'none')}\nTarget scaling used: {model_data.get('target_scaling_type', 'none')}\n")

        elif task == "classification":
            target_encoder = model_data.get('target_encoder')
            y_test_encoded = target_encoder.transform(y_test) if target_encoder else y_test
            class_names = target_encoder.classes_ if target_encoder else None

            y_pred_encoded = model.predict(X_test_scaled)

            click.echo(f"Scaling used: {model_data.get('scaling_type', 'none')}\n")

        if not metrics:
            raise click.ClickException("No metrics provided for evaluation. Use --metrics option to specify metrics.")

    except FileNotFoundError as exc:
        raise click.ClickException(f"File not found: {exc}") from exc
    
    except ValueError as e:
        raise click.ClickException(f"Value error: {e}") from e

    if task == "regression":
        if metrics:
            show_metrics_regression(metrics, y_test, y_pred_unscaled)
            
        if plot:
            plot_predictions_regression(y_test, y_pred_unscaled)

    elif task == "classification":
        if metrics:
            show_metrics_classification(metrics, y_test_encoded, y_pred_encoded)

        if plot:
            plot_predictions_classification(y_test_encoded, y_pred_encoded, class_names)

if __name__ == "__main__":
    evaluate()
