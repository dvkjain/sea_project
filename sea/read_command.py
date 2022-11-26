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

        filename_with_extension = filepath.split("/")[-1]

        if "/" in filepath and "~" in filepath:

            filepath = f"{os.getcwd()}{filename_with_extension}"

        if sheetname != None:

            data = pd.read_excel(f"{filepath}", engine = "openpyxl", sheet_name = sheetname)

        else:

            data = pd.read_excel(f"{filepath}", engine = "openpyxl")
            
        click.echo(data.head())
