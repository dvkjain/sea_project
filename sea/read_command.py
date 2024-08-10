from pathlib import Path
import click
import pandas as pd

class Code:

    def __init__(self):
        pass
    
    @staticmethod
    @click.command()
    @click.argument("filepath")
    @click.option("--sheetname", "-s", help = "Name of the sheet in the spreadsheet. If not given, the first sheet will be considered.")
    def read(filepath, sheetname):

        "Allows the user to read a spreadsheet through a terminal window."

        filepath = Path(filepath).expanduser().resolve()

        try:
            
            if sheetname:
                data = pd.read_excel(filepath, engine="openpyxl", sheet_name=sheetname)

            else:
                data = pd.read_excel(filepath, engine="openpyxl")
            
            click.echo(data.head())

        except Exception as e:
            click.echo(f"Error reading '{filepath}': {e}", err=True)
            