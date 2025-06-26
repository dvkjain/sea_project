import click
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

@click.command()
@click.argument("path")
@click.argument("kind", type=click.Choice(["line", "scatter", "bar", "box", "violin"]))
@click.option("-x", help="Column name for x-axis. If not provided, the first column is used.")
@click.option("-y", help="Column name for y-axis. If not provided, the second column is used.")
@click.option("--sheetname", "--sheet", help="Name of the sheet in the .xlsx or HDF5 file. If not given, the first sheet will be considered.")
@click.option("--save", "-s", help="If provided, the plot will be saved in a .png file with the name given.")
def plot(path, kind, x, y, sheetname, save):

    """Allows the user to plot a spreadsheet through a terminal window.

    Types of plot supported: 'line', 'scatter', 'bar', 'box', 'violin'.

    NOTE: The X and Y axis MUST be COLUMN NAMES in the spreadsheet.
    """

    import seaborn as sns

    path = Path(path).expanduser().resolve()

    try:
        path = str(path)
        if path.endswith('.xlsx'):
            data = pd.read_excel(path, engine="openpyxl", sheet_name=sheetname) if sheetname else pd.read_excel(path, engine="openpyxl")
        elif path.endswith('.csv'):
            data = pd.read_csv(path)
        elif path.endswith('.json'):
            data = pd.read_json(path)
        elif path.endswith('.h5') or path.endswith('.hdf5'):
            data =  pd.read_hdf(path, sheet_name=sheetname) if sheetname else pd.read_hdf(path)
        else:
            raise ValueError("Unsupported file format. Please use .xlsx, .csv, .json, or .h5/.hdf5")
        
        if data.shape[1] < 2:
            raise click.ClickException("Error: The dataset must have at least two columns.")

        # If x or y is missing, it will use the first two columns
        if not x and not y:
            x, y = data.columns[:2]
            click.echo(f"X and Y columns were not specified. Using '{x}' for X and '{y}' for Y.")

        elif not x:
            x = data.columns[0]
            click.echo(f"X column not specified. Using '{x}' for X.")

        elif not y:
            y = data.columns[1]
            click.echo(f"Y column not specified. Using '{y}' for Y.")

    except FileNotFoundError as exc:
        raise click.ClickException(f"File not found: {exc}") from exc
    
    except Exception as e:
        raise click.ClickException(f"Error processing file: {e}") from e

    if x not in data.columns:
        raise click.ClickException(f"Error: Column '{x}' not found in the dataset.")

    if y not in data.columns:
        raise click.ClickException(f"Error: Column '{y}' not found in the dataset.")
    
    plt.figure(figsize=(10, 5))

    plot_func = {
        "line": sns.lineplot,
        "scatter": sns.scatterplot,
        "bar": sns.barplot,
        "box": sns.boxplot,
        "violin": sns.violinplot,
    }.get(kind)

    if plot_func:
        plot_func(data=data, x=x, y=y)

    plt.title(f"{kind.capitalize()} Plot of {y} vs {x}")
    plt.xticks(rotation=45)

    if save:
        plt.savefig(save)
        click.echo(f"Plot saved as '{save}'.")
    
    plt.show()

if __name__ == "__main__":
    plot()
