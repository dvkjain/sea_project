import click
from pathlib import Path
import pandas as pd

@click.command()
@click.argument("filepath")
@click.option("--sheetname", "--sheet", help="Name of the sheet in the spreadsheet. If not given, the first sheet will be considered.")
def read(filepath, sheetname):

    """Allows the user to read a sample of the spreadsheet."""

    filepath = Path(filepath).expanduser().resolve()

    try:
        if sheetname:
            data = pd.read_excel(filepath, engine="openpyxl", sheet_name=sheetname)
        else:
            data = pd.read_excel(filepath, engine="openpyxl")
            
        click.echo(data.head().to_string())

    except FileNotFoundError as exc:
        raise click.ClickException(f"File '{filepath}' not found.") from exc
    
    except Exception as e:
        raise click.ClickException(f"Error processing '{filepath}': {e}") from e

if __name__ == "__main__":
    read()
