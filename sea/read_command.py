import click
from pathlib import Path
import pandas as pd

@click.command()
@click.argument("path")
@click.option("--sheetname", "--sheet", help="Name of the sheet in the .xlsx or HDF5 file. If not given, the first sheet will be considered.")
def read(path, sheetname):

    """Allows the user to read a sample of the spreadsheet."""

    path = Path(path).expanduser().resolve()

    try:

        path = str(path)
        if path.endswith('.xlsx'):
            data =  pd.read_excel(path, engine="openpyxl", sheet_name=sheetname) if sheetname else pd.read_excel(path, engine="openpyxl")
        elif path.endswith('.csv'):
            data =  pd.read_csv(path)
        elif path.endswith('.json'):
            data =  pd.read_json(path)
        elif path.endswith('.h5') or path.endswith('.hdf5'):
            data =  pd.read_hdf(path, sheet_name=sheetname) if sheetname else pd.read_hdf(path)
        else:
            raise ValueError("Unsupported file format. Please use .xlsx, .csv, .json, or .h5/.hdf5")
        
        click.echo(data.head().to_string())

    except FileNotFoundError as exc:
        raise click.ClickException(f"File '{path}' not found.") from exc
    
    except Exception as e:
        raise click.ClickException(f"Error processing '{path}': {e}") from e

if __name__ == "__main__":
    read()
