# Advanced Deep Learning Midterm Project

Create a model with specified hyperparameter grid to train on KMNIST data set. Uses Grid Search and 5-Fold Cross Validation to determine best hyperparameters, then writes model weights, and other training metrics, to the models directory.

## Run Instructions

1. In main.py, set up ARGS global variable.

    * select desired optimizer.
    * specify desired optimizer parameter grid.
    * specify desired training parameter grid.

2. Run main.py

3. Model weights, loss per epoch, accuracy per epoch and training time will be written to models directory.

## Directory Structure

data/
* store KMNIST dataset files here

models/ 
* save model files here

src/ 
* source files

notebooks/ 
* .ipynb files you wish to run, also save graphs here
