import click

from .edit_command import edit
from .read_command import read
from .plot_command import plot
from .ml.train_command import train
from .ml.evaluate_command import evaluate

@click.group()
def cli():
    
    """sea is a command line tool that allows spreadsheet manipulation, editing, plotting, neural network training and evaluating through a terminal window.
    """

cli.add_command(edit)
cli.add_command(read)
cli.add_command(plot)
cli.add_command(train)
cli.add_command(evaluate)