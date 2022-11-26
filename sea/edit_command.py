from pathlib import Path
import click
import openpyxl
import os

class Code:

    def __init__(self):
        pass
    
    @staticmethod
    @click.command()
    @click.argument("filepath")
    @click.option("--sheetname", "-s", help = "Name of the sheet in the spreadsheet. If not given, the active spreadsheet will be considered.")
    @click.argument("cell")
    @click.argument("value")
    @click.option("--new_filename", "-n", help = "Name of the new file where the changes made will be present. If not inserted, changes will appear in the same file.")
    def edit(filepath, sheetname, cell, value, new_filename):

        "Allows simple editing of a spreadsheet."

        xlsx_file = Path(filepath)
        wb = openpyxl.load_workbook(xlsx_file)
        filename_with_extension = filepath.split("/")[-1]
        filename_type = filename_with_extension.split(".")[1]

        if "/" in filepath and "~" in filepath:

            filepath = f"{os.getcwd()}{filename_with_extension}"
            filepath = filepath.split("/")

        # if "/" not in filepath:

        #     filepath = filename_with_extension

        filename = filename_with_extension.split(".")[0]

        if sheetname == None:

            spreadsheet = wb.active
        
        else:

            spreadsheet = wb[sheetname]
            
        spreadsheet[cell] = value

        if new_filename == None:

            new_filename = f"{filename}.{filename_type}"
            wb.save(new_filename)
