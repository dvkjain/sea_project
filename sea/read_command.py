import click

@click.command()
@click.argument("filepath")
@click.option("--sheetname", "--sheet", help="Name of the sheet in the spreadsheet. If not given, the first sheet will be considered.")
def read(filepath, sheetname):

    """Allows the user to read a sample of the spreadsheet."""

    from pathlib import Path
    import pandas as pd

    filepath = Path(filepath).expanduser().resolve()

    try:
        if sheetname:
            data = pd.read_excel(filepath, engine="openpyxl", sheet_name=sheetname)
        else:
            data = pd.read_excel(filepath, engine="openpyxl")
            
        click.echo(data.head().to_string())

    except FileNotFoundError as exc:
        click.echo(f"Error: File '{filepath}' not found.", err=True)
        raise SystemExit(1) from exc
    
    except Exception as e:
        click.echo(f"Error reading '{filepath}': {e}", err=True)
        raise SystemExit(1) from e

if __name__ == "__main__":
    read()
