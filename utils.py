from pathlib import Path

import pandas as pd


def check_dataset_validity(dataset_path: Path):
    """
    Check if the dataset is valid.
    :param dataset_path: Path to the dataset.
    :return: True if valid, False otherwise.
    """
    if not dataset_path.exists():
        print("Dataset path does not exist.")
        return False

    folder = ["train", "valid", "test"]
    csvs = ["annotated_train.csv", "annotated_valid.csv", "annotated_test.csv"]

    # Check if the dataset folder contains the required subfolders and CSV files
    for i in folder + csvs:
        if not (dataset_path / i).exists():
            print(f"'{i}' file does not exist.")
            return False

    for i in csvs:
        data = pd.read_csv(dataset_path / i)

        # Check if the CSV files have the correct columns
        if data.columns != ["img_name","img_path","class_idx","class_name"]:
            print(f"'{i}' file does not have the correct columns.\n"
                  f"Expected: ['img_name', 'img_path', 'class_idx', 'class_name']\n")
            return False

        # Check if the images exist and if class_idx is an integer
        for idx, row in data.iterrows():
            if not (dataset_path / row["img_path"]).exists():
                print(f"Image '{row['img_path']}' in '{i}' file does not exist.")
                return False

            if not isinstance(row["class_idx"], int):
                print(f"Class index in '{i}' file is not an integer.")
                return False

    return True
