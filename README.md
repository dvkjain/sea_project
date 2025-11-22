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

With the train command, supported files can be used to train regression and classification (binary or multiclass) neural network models, using sklearn. The target variable must always be the last column.  
The train command supports .yaml configuration files.  
The default model task is "regression". If you want it changed to "classification", just put "--task classification", or state in your .yaml file 'task: classification'.  
In sklearn, the MLPRegressor is built only with the MSE (Mean Squared Error) loss function type, together with L2 regularization, without any flexibility to change.  
As for MLPClassifier, it is built with Cross-Entropy Loss and L2 regularization, like MLPRegressor. This means the loss function can't be changed here either.

- Supported scalers: MinMax scaling, Standard scaling, and logarithmic scaling.
- Supported activation functions: ReLU, tanh, logistic (sigmoid), and identity.

There is also hyperparameter tuning support! To use it, just use the --auto option, and specify the number of trials (the number of trials you want to run) to let the model search for the best hyperparameters automatically, using RandomizedSearchCV. The cross-validation is set to 5. The hidden layers configuration tuples it compares are: (2), (4), (2, 4), (2, 2), (2, 4, 2).

NOTE 1: if a certain model specification is both in the .yaml file and stated in the terminal, the value stated in the terminal will ALWAYS override the configuration file.  
NOTE 2: if chosen to use hyperparemeter tuning, there is a high chance many warnings will be displayed (e.g. convergence warnings). These warnings can be ignored.  
NOTE 3: hyperparemeter tuning and saving the model can't be done from the .yaml configuration file. If needed, these actions must be stated in the terminal.  
NOTE 4: if chosen to encode (for classification problems), label encoding will be used for the target variable. If chosen to encode in a regression task, a warning will be shown, but the program will ignore it and keep going.  
NOTE 5: target scaling not available for classification tasks. If target scaling is given in these scenarios, a warning will be shown, but it will be ignored by the program, just like with the encoding situation (NOTE 5).    
NOTE 6: if chosen to save, a file will be saved as a dictionary containing the model, its type, training data, scalers, and encoder, if used.  
NOTE 7: if the user states the --auto option (hyperparemeter tuning), most of the arguments WON'T be used by the program (EVEN IF GIVEN by the user), because the tuner will search for these itself: 
- Batch size
- Neurons per layer
- Learning rate
- Optimizer
- Activation funtion

### Examples

```console
sea train .\training_data.xlsx --config configuration.yaml --epochs 200 # Overrides epochs in config

sea train .\training_data.xlsx --task classification --auto 20 --scaling log --encode

sea train .\training_data.xlsx --auto 20 --scaling log --target_scaling standard --save regr_model.joblib

sea train .\training_data.csv --config configuration.yaml --save mlmodel.joblib

sea train .\training_data.xlsx --task classification --encode --save classif_model.joblib
```

### YAML configuration file example

> ```yaml
> train:
>   random_state: 42
>   task: regression
>   scaling: none
>   target_scaling: none
>   epochs: 100
>   batch_size: 32
>   neurons_per_layer: "1"
>   learning_rate: 0.01
>   optimizer: adam
>   activation: relu
> ```

Note that the neurons_per_layer variable HAS to be a string

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
