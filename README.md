# Sea project

The sea cli-tool is a simple way to read, modify, plot graphs, and do much more with spreadsheets using the terminal.

## Read command

With the read command, .xlsx files can be read (only strings/integers).

### Examples

```console
sea read .\spreadsheet.xlsx

sea read .\spreadsheet.xlsx --sheetname Sheet1
```

## Edit command

With the edit command, .xlsx files can be edited (only strings/integers).

### Examples

```console
sea edit .\spreadsheet.xlsx B5 newvalue

sea edit .\spreadsheet.xlsx A2 newvalue --sheetname Sheet1

sea edit .\spreadsheet.xlsx A2 newvalue --sheetname Sheet1 --new_file newdata.xlsx
```

## Plot command

With the plot command, .xlsx files can be plotted into a graph.
Supported graph types: line plot, scatter plot, bar plot, box plot, and violin plot.

NOTE: if at least one of the X and Y columns are not given, the first two columns in the spreadsheet will be used instead.

### Examples

```console
sea plot .\spreadsheet.xlsx line

sea plot .\spreadsheet.xlsx bar -x X_values -y Y_values

sea plot .\spreadsheet.xlsx bar -x X_values -y Y_values --save plotimg.png
```

## Neural command

With the neural command, .xlsx and .csv files can be used to train and evaluate basic neural network models, using sklearn.
The model (and scalers, if used) can also be saved in a joblib file. The target variable must always be the last column.

Supported scalers: MinMax scaling, Standard scaling, and logarithmic scaling.
Supported activation functions: ReLU, tanh, logistic (sigmoid), and identity.

NOTE: if chosen to save the model, it will be saved in a dictionary containing the model and scalers, if given and supported (log scalers are not supported).

### Examples

```console
sea neural .\spreadsheet.xlsx --random_state 42 --activation logistic

sea neural .\spreadsheet.xlsx --random_state 42 --epochs 150 --activation logistic

sea neural .\spreadsheet.xlsx --random_state 42 --save model.joblib
```

## Installation

1. Clone the repository
2. Enter program folder
3. Run the install.sh file
