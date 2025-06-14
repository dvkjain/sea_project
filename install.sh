#!/bin/bash

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python from https://www.python.org/downloads/ and try again."
    exit 1
fi

# Install the package
python -m pip install .