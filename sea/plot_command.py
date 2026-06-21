import click
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

@click.command()
@click.argument("path")
@click.argument("kind", type=click.Choice(["line", "scatter", "bar", "box", "violin"]))
@click.option("-x", help="Column name for x-axis. If not provided, it may be inferred.")
@click.option("-y", help="Column name for y-axis. If not provided, it may be inferred.")
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
            data = pd.read_hdf(path, key=sheetname) if sheetname else pd.read_hdf(path)
        else:
            raise click.ClickException("Unsupported file format. Please use .xlsx, .csv, .json, or .h5/.hdf5")
        
        if data.shape[1] < 1:
            raise click.ClickException("Error: The dataset is empty.")

        # If NEITHER x nor y is provided, default to the first two columns (if possible)
        if not x and not y:
            if data.shape[1] >= 2:
                x, y = data.columns[:2]
                click.echo(f"X and Y columns were not specified. Using '{x}' for X and '{y}' for Y.")
            else:
                y = data.columns[0]
                click.echo(f"X and Y columns were not specified. Only one column found, using '{y}' for Y.")

    except FileNotFoundError as exc:
        raise click.ClickException(f"File not found: {exc}") from exc
    
    except Exception as e:
        raise click.ClickException(f"Error processing file: {e}") from e

    # Validation must check if x/y exist ONLY if they are not None
    if x and x not in data.columns:
        raise click.ClickException(f"Error: Column '{x}' not found in the dataset.")

    if y and y not in data.columns:
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

        plot_args = {"data": data}
        if x: 
            plot_args["x"] = x
        if y: 
            plot_args["y"] = y
            
        # If one is missing, explicitly map it to the DataFrame index.
        if kind in ["line", "scatter"]:
            if x and not y:
                plot_args["y"] = data.index
            elif y and not x:
                plot_args["x"] = data.index

        plot_func(**plot_args)

    # Dynamic title so it reads well whether 1 or 2 variables are used
    title_str = f"{kind.capitalize()} Plot"
    if x and y:
        title_str += f" of {y} vs {x}"
    elif y:
        title_str += f" of {y}"
    elif x:
        title_str += f" of {x}"
        
    plt.title(title_str)
    plt.xticks(rotation=45)

    if save:
        plt.savefig(save)
        click.echo(f"Plot saved as '{save}'.")
    
    plt.show()

if __name__ == "__main__":
    plot()