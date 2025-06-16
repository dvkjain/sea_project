# Sea project

The sea CLI-tool is a simple way to read, modify, plot graphs, and do much more with spreadsheets using the terminal.

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

NOTE: if one of the X and Y columns are not given, the first columns in the spreadsheet will be used instead (first column for X and second column for Y)

### Examples

```console
sea plot .\spreadsheet.xlsx line

sea plot .\spreadsheet.xlsx bar -x X_values -y Y_values

sea plot .\spreadsheet.xlsx bar -x X_values -y Y_values --save plotimg.png
```

## Train command

With the train command, .xlsx and .csv files can be used to train and neural network models, using sklearn. The target variable must always be the last column.

Supported scalers: MinMax scaling, Standard scaling, and logarithmic scaling. Supported activation functions: ReLU, tanh, logistic (sigmoid), and identity.

NOTE: if chosen to save, a file will be saved as a dictionary containing the model, its type, training data, and scalers, if used.

### Examples

```console
sea train .\training_data.xlsx --activation logistic -- scaling minmax

sea train .\training_data.xlsx --epochs 150 --activation logistic

sea train .\training_data.xlsx --save model.joblib
```

## Evaluate command

With the evaluate command, the trained model saved in a .joblib file with the train command can be evaluated using different metrics.

Supported metrics: 
- mse 
- mae 
- mape 
- r2 
- medae 
- evs 
- max error 
- msle

### Examples

```console
sea evaluate .\model.joblib .\test_data.xlsx --metrics mae

sea evaluate .\model.joblib .\test_data.xlsx --metrics r2,mape

sea evaluate .\model.joblib .\test_data.xlsx --metrics "mse, r2"
```

## Installation

1. Clone the repository
2. Enter program folder
3. Run the install.sh file
