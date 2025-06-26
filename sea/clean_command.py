import click
import pandas as pd
from pathlib import Path

@click.command()
@click.argument("path")
@click.option("--dropna", is_flag=True, help="Remove rows with any N/A values.")
@click.option("--drop-duplicates", is_flag=True, help="Remove duplicated rows.")
@click.option("--sheetname", "--sheet", help="Sheet/key name for Excel/HDF5 files.")
@click.option("--save", "-s", help="Output file name. If not provided, will overwrite the original file.")
def clean(path, dropna, drop_duplicates, sheetname, save):
    """
    Cleans a data file (xlsx, csv, json, h5) by removing N/A and/or duplicate rows.
    """
    path = Path(path).expanduser().resolve()
    ext = path.suffix.lower()
    save = save or path

    if ext == ".xlsx":
        df = pd.read_excel(path, engine="openpyxl", sheet_name=sheetname) if sheetname else pd.read_excel(path, engine="openpyxl")
    elif ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".json":
        df = pd.read_json(path)
    elif ext in [".h5", ".hdf5"]:
        if not sheetname:
            raise click.ClickException("For HDF5 files, you must specify --sheetname (the key/dataset name).")
        df = pd.read_hdf(path, key=sheetname)
    else:
        raise click.ClickException("Unsupported file format. Supported: .xlsx, .csv, .json, .h5, .hdf5")

    # Clean data
    if dropna:
        df = df.dropna()
    if drop_duplicates:
        df = df.drop_duplicates()

    # Save cleaned data
    if ext == ".xlsx":
        df.to_excel(save, index=False)
    elif ext == ".csv":
        df.to_csv(save, index=False)
    elif ext == ".json":
        df.to_json(save, orient="records", lines=False)
    elif ext in [".h5", ".hdf5"]:
        df.to_hdf(save, key=sheetname, mode='w')

    if save==path:
        if click.confirm("Warning: The original file will be overwritten with the new changes. Do you want to proceed?"):
            click.echo(f"Cleaned data saved to original file {path}")
        else:
            click.echo("Aborted")
    else:
        click.echo(f"Cleaned data saved to {save}")

if __name__ == "__main__":
    clean()
