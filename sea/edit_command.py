from pathlib import Path
import click
import openpyxl

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

        if sheetname:
            sheet = wb[sheetname]

        else:
            sheet = wb.active
        
        sheet[cell] = value

        if not new_filename:
            new_filename = filepath
        
        else:

            if new_filename == filepath:

                click.echo("Warning: the original file will be overwritten with the new changes, proceed?")
                user = input("Y/N\n")

                if user.lower().strip() == "y":
                    pass

                else:
                    click.echo("Cancelled.")
                    exit()

        try:
            wb.save(new_filename)
            click.echo(f"Changes saved to {new_filename}")

        except Exception as e:
            click.echo(f"Error saving changes: {e}", err=True)
