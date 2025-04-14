import click

@click.command()
@click.argument("filepath")
@click.argument("kind", type=click.Choice(["line", "scatter", "bar", "box", "violin"]))
@click.option("-x", help="Column name for x-axis. If not provided, the first column is used.")
@click.option("-y", help="Column name for y-axis. If not provided, the second column is used.")
@click.option("-sheetname", "--sheet", help="Name of the sheet in the spreadsheet. If not given, the first sheet will be considered.")
@click.option("--save", "-s", help="If provided, the plot will be saved in a .png file with the name given.")
def plot(filepath, kind, x, y, sheetname, saveimg):

    """Allows the user to plot a spreadsheet through a terminal window.

    Types of plot supported: 'line', 'scatter', 'bar', 'box', 'violin'.

    NOTE: The X and Y axis MUST be COLUMN NAMES in the spreadsheet.
    """

    import pandas as pd
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns

    filepath = Path(filepath).expanduser().resolve()

    try:
        if sheetname:
            data = pd.read_excel(filepath, engine="openpyxl", sheet_name=sheetname) 
        else:
            data = pd.read_excel(filepath, engine="openpyxl")

        if data.shape[1] < 2:
            click.echo("Error: The dataset must have at least two columns.", err=True)
            raise SystemExit(1)

        # If x or y is missing, it will use the first two columns
        if not x or not y:
            x, y = data.columns[:2]
            click.echo(f"One or both columns were not specified. Using '{x}' for X and '{y}' for Y.")

    except FileNotFoundError as exc:
        click.echo(f"Error: File '{filepath}' not found.", err=True)
        raise SystemExit(1) from exc
    
    except Exception as e:
        click.echo(f"Error processing '{filepath}': {e}", err=True)
        raise SystemExit(1) from e

    if x not in data.columns or y not in data.columns:
        click.echo(f"Error: Columns '{x}' and/or '{y}' not found in the dataset.", err=True)
        raise SystemExit(1)

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
    else:
        click.echo(f"Error: Unsupported plot kind '{kind}'.", err=True)
        raise SystemExit(1)

    plt.title(f"{kind.capitalize()} Plot of {y} vs {x}")
    plt.xticks(rotation=45)

    if saveimg:
        plt.savefig(saveimg)
        click.echo(f"Plot saved as '{saveimg}'.")
    
    plt.show()

if __name__ == "__main__":
    plot()
