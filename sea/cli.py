import click

from .edit_command import edit
from .read_command import read
from .plot_command import plot
from .neural_command import neural

@click.group()
def cli():
    
    """sea is a command line tool that allows basic spreadsheet manipulation, plotting, and neural network training through a terminal window.
    """

cli.add_command(edit)
cli.add_command(read)
cli.add_command(plot)
cli.add_command(neural)