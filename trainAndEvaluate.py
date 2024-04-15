import torch
# from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from processData import load_and_partition_data
from sklearn.metrics import accuracy_score
import pandas as pd
from simpletransformers.classification import ClassificationModel
import matplotlib.pyplot as plt 

def k_fold_cross_validation(X_train, y_train, model_params):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=486)
    results = []

    for train_idx, val_idx in kf.split(X_train, y_train):
        train_df = pd.DataFrame({'text': X_train.iloc[train_idx], 'labels': y_train.iloc[train_idx]})
        val_df = pd.DataFrame({'text': X_train.iloc[val_idx], 'labels': y_train.iloc[val_idx]})

        model = ClassificationModel('bert', 'bert-base-uncased', num_labels=2, use_cuda=True, args=model_params)  # Adjust parameters as needed
        model.train_model(train_df)

        result, _, _ = model.eval_model(val_df, False, acc=accuracy_score)
        # print(result['acc'])
        results.append(result['acc'])

    return sum(results) / len(results)


def train_and_validate(model, X_train, y_train, X_val, y_val, train_args):
    train_df = pd.DataFrame({'text': X_train, 'labels': y_train})
    val_df = pd.DataFrame({"text": X_val, "labels": y_val})

    # Define a function to capture evaluation metrics during training
    training_history = []
    def capture_metrics(eval_result):
        training_history.append(eval_result)

    model.train_model(train_df, eval_df=val_df, evaluator="sklearn", callbacks=[capture_metrics], args=train_args)

    # Step 3: Plot the training history
    train_loss = [metrics['eval_loss'] for metrics in training_history]
    train_acc = [metrics['eval_accuracy'] for metrics in training_history]
    val_loss = [metrics['eval_loss'] for metrics in training_history]
    val_acc = [metrics['eval_accuracy'] for metrics in training_history]
    steps = [i * train_args['evaluate_during_training_steps'] for i in range(len(training_history))]

    plt.plot(steps, train_loss, label='Training Loss')
    plt.plot(steps, val_loss, label='Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('plots/loss_plot_lr_' + str(train_args['learning_rate']) + '_wd_' + str(train_args['weight_decay']) + '.png')  # Save the plot as loss_plot.png
    #plt.show()

    plt.plot(steps, train_acc, label='Training Accuracy')
    plt.plot(steps, val_acc, label='Validation Accuracy')
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('plots/accuracy_plot_lr_' + str(train_args['learning_rate']) + '_wd_' + str(train_args['weight_decay']) + '.png')  # Save the plot as accuracy_plot.png
    #plt.show()


def validate(model, X_validation, y_validation):
    validation_df = pd.DataFrame({'text': X_validation, 'labels': y_validation})

    result, model_outputs, wrong_predictions = model.eval_model(validation_df, False, acc=accuracy_score)

    return result, model_outputs, wrong_predictions


def test(model, X_test, y_test):
    test_df = pd.DataFrame({'text': X_test, 'labels': y_test})

    result, model_outputs, wrong_predictions = model.eval_model(test_df, False, acc=accuracy_score)

    return result, model_outputs, wrong_predictions

def get_hyperparameters():
    params = {"learning_rates": [10**-j for j in range(3,6)],
              "weight_decays": [0.01, 0.001]}
    return params

def main():
    X_train, X_validation, X_test, y_train, y_validation, y_test = load_and_partition_data("allTweets.csv")

    # learning rates
    for lr in get_hyperparameters()["learning_rates"]:
        for wd in get_hyperparameters()["weight_decays"]: 
            print("lr", lr, "|", "wd", wd)
            model_params = {
                'num_train_epochs': 3,  # Number of training epochs
                'train_batch_size': 32,  # Batch size for training
                'learning_rate': lr,   # Learning rate for optimizer
                'weight_decay': wd,    # Weight decay for regularization
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
                'use_cuda': True,  # Use GPU if available
                'dropout': 0.1,  # Dropout probability for dropout layers
            }

            # model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
            # tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')

            # k_fold_cross_validation(X_train, y_train, model, tokenizer)

            # evaluate(X_train, X_test, y_train, y_test, model, tokenizer)

            # print("Mean accuracy from k-folds:", k_fold_cross_validation(X_train, y_train, model_params))

            model = ClassificationModel('bert', 'bert-base-uncased', num_labels=2, use_cuda=True, args=model_params)

            train_and_validate(model=model, X_train=X_train, y_train=y_train, X_val=X_validation, y_val=y_validation, train_args=model_params)
            #train(model, X_train, y_train)
            result, model_outputs, wrong_predictions = validate(model, X_validation, y_validation)

            result2, model_outputs2, wrong_predictions2 = test(model, X_test, y_test)

            '''print(result)
            print(result2)'''
            # print(model_outputs)
            # print(wrong_predictions)
            with open(f"results/tuning_result_lr_{lr}_wd_{wd}.txt", "w", encoding='utf-8') as outfile:
                outfile.writelines(f"lr: {lr}, wd: {wd}")
                outfile.writelines(f"Validation: {result}")
                outfile.writelines(f"Test: {result2}")


if __name__ == "__main__":
    main()
