import click
import pandas as pd
from pathlib import Path

@click.command()
@click.argument("path")
@click.option("--ratio", help="Ratio of data to use for training.", default=0.6, type=float)
@click.option("--sheetname", "--sheet", help="Sheet/key name for Excel/HDF5 files.")
def split(path, ratio, sheetname):
    """
    Splits a data file (xlsx, csv, json, h5) into training and test sets.
    """
    path = Path(path).expanduser().resolve()
    ext = path.suffix.lower()

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

    
    # Split data into train and test sets
    train_df = df.sample(frac=ratio, random_state=42)
    test_df = df.drop(train_df.index)

    train_ending = "_train" + ext
    test_ending = "_test" + ext
    
    # Save train and test sets
    if ext == ".xlsx":

        train_df.to_excel(path.with_name(path.stem + train_ending), index=False)
        test_df.to_excel(path.with_name(path.stem + test_ending), index=False)
    elif ext == ".csv":
        train_df.to_csv(path.with_name(path.stem + train_ending), index=False)
        test_df.to_csv(path.with_name(path.stem + test_ending), index=False)
    elif ext == ".json":
        train_df.to_json(path.with_name(path.stem + train_ending), orient="records", lines=False)
        test_df.to_json(path.with_name(path.stem + test_ending), orient="records", lines=False)
    elif ext in [".h5", ".hdf5"]:
        train_df.to_hdf(path.with_name(path.stem + train_ending), key=sheetname, mode='w')
        test_df.to_hdf(path.with_name(path.stem + test_ending), key=sheetname, mode='w')


if __name__ == "__main__":
    split()
