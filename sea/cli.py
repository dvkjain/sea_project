import click

from .clean_command import clean
from .read_command import read
from .plot_command import plot
from .ml.split_command import split
from .ml.train_command import train
from .ml.evaluate_command import evaluate

@click.group()
def cli():
    
    """sea is a command line tool that allows spreadsheet manipulation (cleaning data, spliting data), plotting, neural network training and evaluating through a terminal window.
    """

cli.add_command(clean)
cli.add_command(read)
cli.add_command(plot)
cli.add_command(split)
cli.add_command(train)
cli.add_command(evaluate)