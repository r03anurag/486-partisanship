import torch
# from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from processData import load_and_partition_data
from sklearn.metrics import accuracy_score
import pandas as pd
from simpletransformers.classification import ClassificationModel
import seaborn as sns
import matplotlib.pyplot as plt 

def k_fold_cross_validation(X_train, y_train, model_params):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=486)
    results = []

    for train_idx, val_idx in kf.split(X_train, y_train):
        train_df = pd.DataFrame({'text': X_train.iloc[train_idx], 'labels': y_train.iloc[train_idx]})
        val_df = pd.DataFrame({'text': X_train.iloc[val_idx], 'labels': y_train.iloc[val_idx]})

        model = ClassificationModel('bert', 'bert-base-uncased', num_labels=2, use_cuda=False, args=model_params)  # Adjust parameters as needed
        model.train_model(train_df)

        result, _, _ = model.eval_model(val_df, False, acc=accuracy_score)
        # print(result['acc'])
        results.append(result['acc'])

    return sum(results) / len(results)


def train(model, X_train, y_train):
    train_df = pd.DataFrame({'text': X_train, 'labels': y_train})
    model.train_model(train_df)


def validate(model, X_validation, y_validation):
    validation_df = pd.DataFrame({'text': X_validation, 'labels': y_validation})

    result, model_outputs, wrong_predictions = model.eval_model(validation_df, False, acc=accuracy_score)

    return result, model_outputs, wrong_predictions


def test(model, X_test, y_test):
    test_df = pd.DataFrame({'text': X_test, 'labels': y_test})

    result, model_outputs, wrong_predictions = model.eval_model(test_df, False, acc=accuracy_score)

    return result, model_outputs, wrong_predictions


def main():
    X_train, X_validation, X_test, y_train, y_validation, y_test = load_and_partition_data("allTweets.csv")

    model_params = {
        'num_train_epochs': 3,  # Number of training epochs
        'train_batch_size': 32,  # Batch size for training
        'learning_rate': 1e-5,   # Learning rate for optimizer
        'weight_decay': 0.01,    # Weight decay for regularization
        'optimizer': "AdamW",
        'overwrite_output_dir': True,  # Overwrite output directory if it exists
        'save_steps': -1,  # Do not save models during training
        'no_cache': True,  # Do not cache features to save memory
        'use_early_stopping': True,  # Use early stopping during training
        'early_stopping_patience': 3,  # Number of epochs to wait before early stopping
        'eval_batch_size': 32,  # Batch size for evaluation
        'max_seq_length': 128,  # Maximum sequence length for input
        'manual_seed': 486,  # Set seed for reproducibility
        'output_dir': 'outputs/',  # Output directory for model checkpoints and predictions
        'cache_dir': 'cache_dir/',  # Directory for storing cache files
        'fp16': False,  # Enable mixed precision training
        'use_cuda': False,  # Use GPU if available
        'dropout': 0.1,  # Dropout probability for dropout layers
    }

    # model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
    # tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')

    # k_fold_cross_validation(X_train, y_train, model, tokenizer)

    # evaluate(X_train, X_test, y_train, y_test, model, tokenizer)

    # print("Mean accuracy from k-folds:", k_fold_cross_validation(X_train, y_train, model_params))

    model = ClassificationModel('bert', 'bert-base-uncased', num_labels=2, use_cuda=False, args=model_params)

    train(model, X_train, y_train)
    result, model_outputs, wrong_predictions = validate(model, X_validation, y_validation)
    result2, model_outputs2, wrong_predictions2 = test(model, X_test, y_test)

    print(result)
    print(result2)
    # print(model_outputs)
    # print(wrong_predictions)


if __name__ == "__main__":
    main()
