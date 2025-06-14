import click

@click.command()
@click.argument("filepath")
@click.argument("cell")
@click.argument("value")
@click.option("--sheetname", "--sheet", help="Name of the sheet in the spreadsheet. If not given, the active sheet will be considered.")
@click.option("--new_file", "-n", help="Name of the new file where changes will be saved. If not provided, changes will be saved to the original file.")
def edit(filepath, cell, value, sheetname, new_file):

    """Allows simple editing of a spreadsheet."""
    
    from pathlib import Path
    import openpyxl

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

        if new_file is None:
            new_file = filepath  # If no new filename provided, overwrite the original file

        if new_file == filepath:
            if click.confirm("Warning: The original file will be overwritten with the new changes. Do you want to proceed?"):
                wb.save(new_file)
                click.echo(f"Changes saved to {new_file}\n")

            else:
                click.echo("Process canceled. No changes were made.\n")

        else:
            wb.save(new_file)
            click.echo(f"Changes saved to {new_file}\n")

    except FileNotFoundError as exc:
        click.echo(f"Error: File '{filepath}' not found.", err=True)
        raise SystemExit(1) from exc
    
    except Exception as e:
        click.echo(f"Error processing '{filepath}': {e}", err=True)
        raise SystemExit(1) from e

if __name__ == "__main__":
    edit()