import click

from .copy_command import Code as copy
from .edit_command import Code as edit
from .read_command import Code as read

@click.group()
def cli():
    
    "sea is a simple command line tool that allows basic spreadsheet manipulation through a terminal window."

cli.add_command(copy.copy)
cli.add_command(edit.edit)
cli.add_command(read.read)
