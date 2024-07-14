import click

from .edit_command import Code as edit
from .read_command import Code as read

@click.group()
def cli():
    
    """sea is a simple command line tool that allows basic spreadsheet manipulation (especially .xlsx) through a terminal window.

       Functionality may be limited if a different format than .xlsx is being used
    """

cli.add_command(edit.edit)
cli.add_command(read.read)
