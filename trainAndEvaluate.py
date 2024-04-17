"""
trainAndEvaluate.py

Trains and determines performance of BERT model.
"""
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from processData import load_and_partition_data
from sklearn.metrics import accuracy_score
import pandas as pd
from simpletransformers.classification import ClassificationModel
import matplotlib.pyplot as plt 


# Not used because chose different method for efficiency
def k_fold_cross_validation(X_train, y_train, model_params):
    """Performs k-fold cross-validation on a classification model.

    Args:
        X_train (pd.Series): Input features for training.
        y_train (pd.Series): Target labels for training.
        model_params (dict): Parameters for the classification model.

    Returns:
        float: Mean accuracy score across all folds.
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=486)
    results = []

    for train_idx, val_idx in kf.split(X_train, y_train):
        train_df = pd.DataFrame({'text': X_train.iloc[train_idx], 'labels': y_train.iloc[train_idx]})
        val_df = pd.DataFrame({'text': X_train.iloc[val_idx], 'labels': y_train.iloc[val_idx]})

        model = ClassificationModel('bert', 'bert-base-uncased', num_labels=2, use_cuda=False, args=model_params)
        model.train_model(train_df)

        result, _, _ = model.eval_model(val_df, False, acc=accuracy_score)
        results.append(result['acc'])

    return sum(results) / len(results)


def train(model, X_train, y_train):
    """
    Trains the classification model.

    Args:
        model (ClassificationModel): Classification model instance.
        X_train (pd.Series): Input features for training.
        y_train (pd.Series): Target labels for training.
    """
    train_df = pd.DataFrame({'text': X_train, 'labels': y_train})
    model.train_model(train_df)


def validate(model, X_validation, y_validation):
    """
    Validates the classification model.

    Args:
        model (ClassificationModel): Classification model instance.
        X_validation (pd.Series): Input features for validation.
        y_validation (pd.Series): Target labels for validation.

    Returns:
        Tuple[Dict, List, List]: Evaluation result, model outputs, and wrong predictions.
    """
    validation_df = pd.DataFrame({'text': X_validation, 'labels': y_validation})
    result, model_outputs, wrong_predictions = model.eval_model(validation_df, False, acc=accuracy_score)
    return result, model_outputs, wrong_predictions


def test(model, X_test, y_test):
    """
    Tests the classification model.

    Args:
        model (ClassificationModel): Classification model instance.
        X_test (pd.Series): Input features for testing.
        y_test (pd.Series): Target labels for testing.

    Returns:
        Tuple[Dict, List, List]: Evaluation result, model outputs, and wrong predictions.
    """
    test_df = pd.DataFrame({'text': X_test, 'labels': y_test})
    result, model_outputs, wrong_predictions = model.eval_model(test_df, False, acc=accuracy_score)
    return result, model_outputs, wrong_predictions
    

def get_hyperparameters():
    """
    Provides hyperparameters for hyperparameter tuning.

    Returns:
        dict: Hyperparameters for hyperparameter tuning.
    """
    params = {
        "learning_rates": [10 ** -i for i in range(3, 6)],
        "weight_decays": [10 ** -i for i in range(1, 6)]
    }

    return params

def main():
    """
    Main function to train and evaluate the classification model.
    """
    X_train, X_validation, X_test, y_train, y_validation, y_test = load_and_partition_data("allTweets.csv")

    # Grid search for hyperparameter tuning
    for lr in get_hyperparameters()["learning_rates"]:
        for wd in get_hyperparameters()["weight_decays"]:

            print(f"lr {lr} | wd {wd}")

            model_params = {
                'num_train_epochs': 3,          # Number of training epochs
                'train_batch_size': 32,         # Batch size for training
                'learning_rate': lr,            # Learning rate for optimizer
                'weight_decay': wd,             # Weight decay for regularization
                'optimizer': "AdamW",           # Adam Optimizer
                'overwrite_output_dir': True,   # Overwrite output directory if it exists
                'save_steps': -1,               # Do not save models during training
                'no_cache': True,               # Do not cache features to save memory
                'use_early_stopping': True,     # Use early stopping during training
                'early_stopping_patience': 3,   # Number of epochs to wait before early stopping
                'eval_batch_size': 32,          # Batch size for evaluation
                'max_seq_length': 128,          # Maximum sequence length for input
                'manual_seed': 486,             # Set seed for reproducibility
                'output_dir': 'outputs/',       # Output directory for model checkpoints and predictions
                'cache_dir': 'cache_dir/',      # Directory for storing cache files
                'fp16': False,                  # Enable mixed precision training
                'use_cuda': False,              # Use GPU if available
                'dropout': 0.1,                 # Dropout probability for dropout layers
            }

            model = ClassificationModel('bert', 'bert-base-uncased', num_labels=2, use_cuda=False, args=model_params)

            train(model, X_train, y_train)

            result, _, _ = validate(model, X_validation, y_validation)

            result2, _, _ = test(model, X_test, y_test)

            with open(f"results/lr_{lr}_wd_{wd}.txt", "w", encoding="utf-8") as outfile:
                outfile.write(f"lr: {lr}, wd: {wd}\n")
                outfile.write(f"Validation: {result}\n")
                outfile.write(f"Test: {result2}\n")

    # Obtaining incorrect predictions for optimal model
    optimal_model_params = {
        'num_train_epochs': 3,          # Number of training epochs
        'train_batch_size': 32,         # Batch size for training
        'learning_rate': 0.0001,        # Learning rate for optimizer
        'weight_decay': 0.0001,         # Weight decay for regularization
        'optimizer': "AdamW",           # Adam Optimizer
        'overwrite_output_dir': True,   # Overwrite output directory if it exists
        'save_steps': -1,               # Do not save models during training
        'no_cache': True,               # Do not cache features to save memory
        'use_early_stopping': True,     # Use early stopping during training
        'early_stopping_patience': 3,   # Number of epochs to wait before early stopping
        'eval_batch_size': 32,          # Batch size for evaluation
        'max_seq_length': 128,          # Maximum sequence length for input
        'manual_seed': 486,             # Set seed for reproducibility
        'output_dir': 'outputs/',       # Output directory for model checkpoints and predictions
        'cache_dir': 'cache_dir/',      # Directory for storing cache files
        'fp16': False,                  # Enable mixed precision training
        'use_cuda': False,              # Use GPU if available
        'dropout': 0.1,                 # Dropout probability for dropout layers
    }

    optimal_model = model = ClassificationModel('bert', 'bert-base-uncased', num_labels=2, use_cuda=False, args=optimal_model_params)

    train(optimal_model, X_train, y_train)

    _, _, wrong_predictions = test(model, X_test, y_test)

    with open("incorrectPredictions.txt", 'w', encoding="utf-8") as outputfile:
        outputfile.write(f"{wrong_predictions}\n")


if __name__ == "__main__":
    main()
