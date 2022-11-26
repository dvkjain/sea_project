import click, shutil, os

class Code:

    def __init__(self):
        pass
    
    @staticmethod
    @click.command()
    @click.argument("filepath")
    def copy(filepath):

        "Makes a copy of specified spreadsheet to be made on currrent working directory."

        filename_with_extension = filepath.split("/")[-1]
        filename = filename_with_extension.split(".")[0]
        filename_type = filename_with_extension.split(".")[1]
        shutil.copy(filepath, (f"{os.getcwd()}{filename}_copy.{filename_type}"))
