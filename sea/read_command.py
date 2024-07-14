import click
import pandas as pd
import os

class Code:

    def __init__(self):
        pass
    
    @staticmethod
    @click.command()
    @click.argument("filepath")
    @click.option("--sheetname", "-s", help = "Name of the sheet in the spreadsheet. If not given, the first sheet will be considered.")
    def read(filepath, sheetname):

        "Allows the user to read a spreadsheet through a terminal window."

        # Check if filepath contains ~ and replace with expanded home directory
        if "~" in filepath:
            filepath = os.path.expanduser(filepath)

        if not os.path.isabs(filepath):
            # Constructs an absolute file path based on the current working directory
            filepath = os.path.join(os.getcwd(), filepath)

        try:
            
            if sheetname:
                data = pd.read_excel(filepath, engine="openpyxl", sheet_name=sheetname)

            else:
                data = pd.read_excel(filepath, engine="openpyxl")
            
            click.echo(data.head())

        except Exception as e:
            click.echo(f"Error reading '{filepath}': {e}", err=True)
            