from pathlib import Path
import click
import openpyxl

@click.command()
@click.argument("filepath")
@click.option("-s", "--sheetname", help="Name of the sheet in the spreadsheet. If not given, the active sheet will be considered.")
@click.argument("cell")
@click.argument("value")
@click.option("-n", "--new_filename", help="Name of the new file where changes will be saved. If not provided, changes will be saved to the original file.")
def edit(filepath, sheetname, cell, value, new_filename):

    """Allows simple editing of a spreadsheet."""

    xlsx_file = Path(filepath)
    
    try:
        wb = openpyxl.load_workbook(xlsx_file)

        if sheetname:
            if sheetname in wb.sheetnames:
                sheet = wb[sheetname]

            else:
                click.echo(f"Error: Sheet '{sheetname}' does not exist.", err=True)
                raise SystemExit(1)
        
        else:
            sheet = wb.active  # Default to the active sheet
        
        sheet[cell] = value

        if new_filename is None:
            new_filename = filepath  # If no new filename provided, overwrite the original file

        if new_filename == filepath:
            if click.confirm("Warning: The original file will be overwritten with the new changes. Do you want to proceed?"):
                wb.save(new_filename)
                click.echo(f"Changes saved to {new_filename}\n")

            else:
                click.echo("Process canceled. No changes were made.\n")
        
        else:
            wb.save(new_filename)
            click.echo(f"Changes saved to {new_filename}\n")

    except Exception as e:
        click.echo(f"Error processing '{filepath}': {e}", err=True)
        raise SystemExit(1)
