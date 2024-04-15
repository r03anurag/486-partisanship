"""Provides functions for loading, preprocessing, tokenizing, and partitioning data."""

import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import os
import re


def aggregate_csvs(outfile_name):
    democrat_folder = "output/democrats/"
    republican_folder = "output/republicans/"

    with open(outfile_name, "w", encoding="utf-8") as output_file:
        output_file.write("Username,Label,Tweet\n")

        for filename in os.listdir(democrat_folder):
            file_path = os.path.join(democrat_folder, filename)

            with open(file_path, 'r', encoding="utf-8") as current_file:
                next(current_file)

                for line in current_file:
                    output_file.write(line)

        for filename in os.listdir(republican_folder):
            file_path = os.path.join(republican_folder, filename)

            with open(file_path, 'r', encoding="utf-8") as current_file:
                next(current_file)

                for line in current_file:
                    output_file.write(line)


def load_and_partition_data(filename):
    """Load data into Pandas dataframe and split into training and testing sets."""

    df = pd.read_csv(filename)

    data_copy = df.copy()

    data_copy["Tweet"] = data_copy["Tweet"].str.lower()
    data_copy["Tweet"] = data_copy["Tweet"].apply(lambda x: re.sub(r"[^a-zA-Z0-9'\"/\-.: ]", "", x))
    # data_copy["Tweet"] = data_copy["Tweet"].apply(lambda x: " ".join(nltk.word_tokenize(x)))

    # Split the data into features (X) and labels (y)
    X = data_copy["Tweet"]
    y = data_copy["Label"]


    X_train, X_validation_and_test, y_train, y_validation_and_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=486)
    X_validation, X_test, y_validation, y_test = train_test_split(X_validation_and_test, y_validation_and_test, test_size=0.5, stratify=y_validation_and_test, random_state=486)

    # # Split the data into training and testing sets with equal label distribution
    # X_train, X_validation_and_test, y_train, y_validation_and_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=486)
    #
    # print(f"Validation and test set size: {len(X_validation_and_test)}")
    # print(f"Validation and test set size: {len(y_validation_and_test)}")
    #
    #
    # X_validation, X_test, y_validation, y_test = train_test_split(X_validation_and_test, y_validation_and_test, test_size=0.5, stratify=y, random_state=486)

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

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def main():
    # aggregate_csvs("allTweets.csv")

    load_and_partition_data("allTweets.csv")


if __name__ == "__main__":
    main()
