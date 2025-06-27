# Sea project

The sea CLI-tool is a simple way to read, clean data, plot graphs, train and evaluate neural networks using the terminal.

Supported file formats:
- .xlsx
- .csv
- .json
- .h5 or .hdf5

## Read command

With the read command, supported files can be read (only strings/integers).

### Examples

```console
sea read .\data.csv

sea read .\data.xlsx --sheetname Sheet1
```

## Clean command

With the clean command, supported files can be cleaned (dropna and remove duplicate rows).

### Examples

```console
sea clean .\data.csv --dropna

sea clean .\data.xlsx  --sheetname Sheet1 --drop-duplicates

sea clean .\data.xlsx --dropna --drop-duplicates --save newdata.xlsx
```

## Plot command

With the plot command, .xlsx files can be plotted into a graph.  
Supported graph types: line plot, scatter plot, bar plot, box plot, and violin plot.

NOTE: if one of the X and Y columns are not given, the first columns in the spreadsheet will be used instead (first column for X and second column for Y)

### Examples

```console
sea plot .\data.csv line

sea plot .\data.xlsx bar -x X_values -y Y_values

sea plot .\data.xlsx bar -x X_values -y Y_values --save plotimg.png
```

## Train command

With the train command, supported files can be used to train regression and classification neural network models, using sklearn. The target variable must always be the last column.  
The default model task is "regression". If you want it changed to "classification", just put "--task classification".  
In sklearn, the MLPRegressor is built only with the MSE (Mean Squared Error) loss function type, together with L2 regularization, without any flexibility to change.  
As for MLPClassifier, it is built with Cross-Entropy Loss and L2 regularization, like MLPRegressor. This means the loss function can't be changed here either.

- Supported scalers: MinMax scaling, Standard scaling, and logarithmic scaling.
- Supported activation functions: ReLU, tanh, logistic (sigmoid), and identity.

NOTE 1: if chosen to encode (for classification problems), label encoding will be used for the target variable.  
NOTE 2: target scaling not available for classification tasks. If target scaling given in these scenarios, it will just be ignored by the program.  
NOTE 3: if chosen to save, a file will be saved as a dictionary containing the model, its type, training data, scalers, and encoder, if used.

### Examples

```console
sea train .\training_data.csv --batch_size all -- scaling minmax

sea train .\training_data.xlsx --epochs 150 --optimizer sgd --activation tanh

sea train .\training_data.xlsx --task classification --encode --save classif_model.joblib
```

## Evaluate command

With the evaluate command, the trained model saved in a .joblib file with the train command can be evaluated using different metrics.

Supported metrics (regression): 
- mse 
- mae 
- mape 
- r2 
- medae 
- evs 
- max error 
- msle

Suported metrics (classification):
- accuracy score
- precision score,
- recall score,
- f1 score,
- roc auc score,
- confusion matrix,
- classification report

NOTE 1: the program automatically checks if the model is a regression or classification type, so there is no --task option, like in the train command.  
NOTE 2: if encoder was used in the training process of a classification model type, the program will automatically apply label encoding for the target variable in the testing data. The same applies to the scalers used.

### Examples

```console
sea evaluate .\regr_model.joblib .\test_data.csv --metrics mae

sea evaluate .\regr_model.joblib .\test_data.xlsx --metrics r2,mape

sea evaluate .\classif_model.joblib .\test_data.xlsx --metrics accuracy,precision
```

## Installation

1. Clone the repository
2. Enter program folder
3. Run the install.sh file
