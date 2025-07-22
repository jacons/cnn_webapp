import json
import os
from pathlib import Path
from typing import List, Union, Dict, Any

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



def find_folders_with_model(folder:str)-> List[Path]:
    """
    Find all folders containing a model file named 'model.pth'.
    :param folder:
    :return:
    """
    matching_folders = []
    for root, dirs, files in os.walk(folder):
        if "model.pth" in files:
            matching_folders.append(root)
    return matching_folders



def save_json(data: Union[Dict, List], filepath: Path, default: Any = None):
    """
    Saves JSON data to a file.

    Parameters:
    -----------
    data : Union[Dict, List]
        JSON data to be saved.

    filepath : Path
        Path object where the JSON data will be saved.

    default : Any, optional
        Default function for serializing objects that cannot be serialized by default, by default None.

    Raises:
    -------
    TypeError
        If the data provided is not serializable to JSON.
    IOError
        If there is an issue writing the file to the specified path.
    """
    with filepath.open(mode="w") as file:
        json.dump(data, file, indent=2, default=default)


def load_json(filepath: Path) -> Union[Dict, List]:
    """
    Loads JSON data from a file.

    Parameters:
    -----------
    filepath : Path
        Path object from where the JSON data will be loaded.

    Returns:
    --------
    Union[Dict, List]
        Loaded JSON data.

    Raises:
    -------
    FileNotFoundError
        If the specified file does not exist.
    JSONDecodeError
        If the file contains invalid JSON.
    IOError
        If there is an issue reading the file from the specified path.
    """
    with filepath.open(mode="r", encoding="utf-8") as file:
        dict_ = json.load(file)
    return dict_