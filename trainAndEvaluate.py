import torch
# from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from processData import load_and_partition_data

# Assuming you have already prepared your data and loaded it into X_train, X_test, y_train, y_test

# Tokenize and preprocess your data
def preprocess_data(texts, labels, tokenizer):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels


def k_fold_cross_validation(X_train, y_train, model, tokenizer):
    X_train_ids, X_train_masks, y_train_tensor = preprocess_data(X_train, y_train, tokenizer)

    # Configure training parameters
    batch_size = 32
    epochs = 3
    learning_rate = 2e-5

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Define k-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_ids, y_train)):
        print(f"Fold {fold + 1}")

        # Create data loaders for training and validation
        train_dataset = TensorDataset(X_train_ids[train_idx], X_train_masks[train_idx], y_train_tensor[train_idx])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(X_train_ids[val_idx], X_train_masks[val_idx], y_train_tensor[val_idx])
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Training loop for this fold
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            # Validation loop for this fold
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = batch
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    total_val_loss += outputs.loss.item()
            avg_val_loss = total_val_loss / len(val_loader)

            print(f"Epoch {epoch + 1}: Validation Loss: {avg_val_loss}")

        # Evaluate on test set after each fold if needed


def evaluate(X_train, X_test, y_train, y_test):

    # Initialize BERT tokenizer and model
    model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')

    k_fold_cross_validation(X_train, y_train, model, tokenizer)

    batch_size = 32

    # After all folds, evaluate on the test set
    # Preprocess test data and create DataLoader
    X_test_ids, X_test_masks, y_test_tensor = preprocess_data(X_test, y_test, tokenizer)
    test_dataset = TensorDataset(X_test_ids, X_test_masks, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_test_loss += outputs.loss.item()
    avg_test_loss = total_test_loss / len(test_loader)

    print(f"Test Loss: {avg_test_loss}")


def main():
    X_train, X_test, y_train, y_test = load_and_partition_data("testData.csv")

    evaluate(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
