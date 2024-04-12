"""Provides functions for loading, preprocessing, tokenizing, and partitioning data."""

import pandas as pd
from sklearn.model_selection import train_test_split
import nltk


def load_and_partition_data(filename):
    """Load data into Pandas dataframe and split into training and testing sets."""

    df = pd.read_csv(filename)

    data_copy = df.copy()

    data_copy["Tweet"] = data_copy["Tweet"].str.lower()
    data_copy["Tokenized_Tweets"] = data_copy["Tweet"].apply(lambda x: nltk.word_tokenize(x))

    # Split the data into features (X) and labels (y)
    X = data_copy["Tweet"]
    y = data_copy["Label"]

    # Split the data into training and testing sets with equal label distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=486)

    # # Print the sizes of the datasets
    # print(f"Training set size: {len(X_train)}")
    # print(f"Testing set size: {len(X_test)}")
    #
    # # Verify the label distribution in the training and testing sets
    # print("Training set label distribution:")
    # print(y_train.value_counts())
    # print("Testing set label distribution:")
    # print(y_test.value_counts())

    # print(X_train)
    # print(y_train)
    # print(X_test)
    # print(y_test)

    return X_train, X_test, y_train, y_test


def main():
    load_and_partition_data("testData.csv")


if __name__ == "__main__":
    main()
