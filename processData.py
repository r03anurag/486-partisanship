"""
processData.py

Provides functions for loading, preprocessing, tokenizing, and partitioning data.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import os
import re


def aggregate_csvs(outfile_name):
    """
    Aggregate data from separate CSV files into a single CSV file.
    Args:
        outfile_name (str): Name of the output CSV file.
    """
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
    """
    Load data into Pandas dataframe and split into training and testing sets.
    Args:
        filename (str): Path to the input CSV file.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]: 
        X_train, X_validation, X_test, y_train, y_validation, y_test
    """

    df = pd.read_csv(filename)

    data_copy = df.copy()

    data_copy["Tweet"] = data_copy["Tweet"].str.lower()
    data_copy["Tweet"] = data_copy["Tweet"].apply(lambda x: re.sub(r"[^a-zA-Z0-9'\"/\-.: ]", "", x))

    # Split the data into features (X) and labels (y)
    X = data_copy["Tweet"]
    y = data_copy["Label"]

    X_train, X_validation_and_test, y_train, y_validation_and_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=486)
    X_validation, X_test, y_validation, y_test = train_test_split(X_validation_and_test, y_validation_and_test, test_size=0.5, stratify=y_validation_and_test, random_state=486)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def main():
    """
    Is entry point for the script.
    """
    # aggregate_csvs("allTweets.csv")  
    # ~~uncomment this^ if this is your first run after pulling data using collectData.py!~~
    load_and_partition_data("allTweets.csv")


if __name__ == "__main__":
    main()
