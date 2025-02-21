import click

from .edit_command import edit
from .read_command import read
from .plot_command import plot

@click.group()
def cli():
    
    """sea is a simple command line tool that allows basic spreadsheet manipulation and plotting (especially .xlsx) through a terminal window.

       Functionality may be limited if a different format than .xlsx is being used
    """

cli.add_command(edit)
cli.add_command(read)
cli.add_command(plot)