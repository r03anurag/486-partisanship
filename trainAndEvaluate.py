import torch
# from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from processData import load_and_partition_data
from sklearn.metrics import accuracy_score
import pandas as pd
from simpletransformers.classification import ClassificationModel

# Assuming you have already prepared your data and loaded it into X_train, X_test, y_train, y_test

# # Tokenize and preprocess your data
# def preprocess_data(texts, labels, tokenizer):
#     input_ids = []
#     attention_masks = []
#     for text in texts:
#         encoded_dict = tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
#         input_ids.append(encoded_dict['input_ids'])
#         attention_masks.append(encoded_dict['attention_mask'])
#     input_ids = torch.cat(input_ids, dim=0)
#     attention_masks = torch.cat(attention_masks, dim=0)
#     labels = torch.tensor(labels)
#
#     return input_ids, attention_masks, labels


# def k_fold_cross_validation(X_train, y_train, model, tokenizer):
#     X_train_ids, X_train_masks, y_train_tensor = preprocess_data(X_train, y_train, tokenizer)
#
#     # Configure training parameters
#     batch_size = 32
#     epochs = 3
#     learning_rate = 2e-5
#
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#
#     # Define k-fold cross-validation
#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_ids, y_train)):
#         print(f"Fold {fold + 1}")
#
#         # Create data loaders for training and validation
#         train_dataset = TensorDataset(X_train_ids[train_idx], X_train_masks[train_idx], y_train_tensor[train_idx])
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
#         val_dataset = TensorDataset(X_train_ids[val_idx], X_train_masks[val_idx], y_train_tensor[val_idx])
#         val_loader = DataLoader(val_dataset, batch_size=batch_size)
#
#         # Training loop for this fold
#         for epoch in range(epochs):
#             model.train()
#             for batch in train_loader:
#                 input_ids, attention_mask, labels = batch
#                 optimizer.zero_grad()
#                 outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#                 loss = outputs.loss
#                 loss.backward()
#                 optimizer.step()
#
#             # Validation loop for this fold
#             model.eval()
#             total_val_loss = 0
#             with torch.no_grad():
#                 for batch in val_loader:
#                     input_ids, attention_mask, labels = batch
#                     outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#                     total_val_loss += outputs.loss.item()
#             avg_val_loss = total_val_loss / len(val_loader)
#
#             print(f"Epoch {epoch + 1}: Validation Loss: {avg_val_loss}")
#
#         # Evaluate on test set after each fold if needed


# def evaluate(X_train, X_test, y_train, y_test, model, tokenizer):
#
#     # k_fold_cross_validation(X_train, y_train, model, tokenizer)
#
#     batch_size = 32
#
#     # After all folds, evaluate on the test set
#     # Preprocess test data and create DataLoader
#     X_test_ids, X_test_masks, y_test_tensor = preprocess_data(X_test, y_test, tokenizer)
#     test_dataset = TensorDataset(X_test_ids, X_test_masks, y_test_tensor)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)
#
#     model.eval()
#     total_test_loss = 0
#     with torch.no_grad():
#         for batch in test_loader:
#             input_ids, attention_mask, labels = batch
#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             total_test_loss += outputs.loss.item()
#     avg_test_loss = total_test_loss / len(test_loader)
#
#     print(f"Test Loss: {avg_test_loss}")


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


def evaluate(model, X_train, X_test, y_train, y_test):
    train_df = pd.DataFrame({'text': X_train, 'labels': y_train})
    test_df = pd.DataFrame({'text': X_test, 'labels': y_test})

    model.train_model(train_df)

    result, model_outputs, wrong_predictions = model.eval_model(test_df, False, acc=accuracy_score)

    return result, model_outputs, wrong_predictions


def main():
    X_train, X_test, y_train, y_test = load_and_partition_data("testData.csv")

    model_params = {
        'num_train_epochs': 10,  # Number of training epochs
        'train_batch_size': 32,  # Batch size for training
        'learning_rate': 1e-5,   # Learning rate for optimizer
        'weight_decay': 0.01,    # Weight decay for regularization
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

    result, model_outputs, wrong_predictions = evaluate(model, X_train, X_test, y_train, y_test)

    print(result)
    print(model_outputs)
    print(wrong_predictions)


if __name__ == "__main__":
    main()
