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
            if sheetname in wb.sheetnames:
                sheet = wb[sheetname]

            else:
                click.echo(f"Error: Sheet '{sheetname}' does not exist.", err=True)
                exit(1)

        else:
            sheet = wb.active
        
        try:
            sheet[cell] = value

        except Exception as e:
            click.echo(f"Error updating cell: {e}", err=True)
            exit(1)

        if new_filename == filepath:
            if click.confirm("Warning: the original file will be overwritten with the new changes, proceed?"):
                try:
                    wb.save(new_filename)
                    click.echo(f"Changes saved to {new_filename}\n")

                except Exception as e:
                    click.echo(f"Error saving changes: {e}", err=True)

            else:
                click.echo("Process canceled, no changes were made.\n")
                exit()
        else:
            try:
                wb.save(new_filename)
                click.echo(f"Changes saved to {new_filename}\n")

            except Exception as e:
                click.echo(f"Error saving changes: {e}", err=True)
