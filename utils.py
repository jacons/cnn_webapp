"""
================================================================================
Author: Andrea Iommi
Code Ownership:
    - All Python source code in this file is written solely by the author.
Documentation Notice:
    - All docstrings and inline documentation are written by ChatGPT,
      but thoroughly checked and approved by the author for accuracy.
================================================================================
"""

import json
import os
from pathlib import Path
from typing import List, Union, Dict, Any

import pandas as pd


def check_dataset_validity(dataset_path: Path) -> bool:
    """
    Validates the structure and integrity of a dataset directory.

    This function checks:
    - Whether the dataset path exists.
    - Whether required subfolders (`train`, `valid`, `test`) and CSV files 
      (`annotated_train.csv`, `annotated_valid.csv`, `annotated_test.csv`) exist.
    - Whether the CSV files contain the correct columns.
    - Whether each listed image file exists in the dataset folder.
    - Whether the `class_idx` column contains only integer values.

    Parameters
    ----------
    dataset_path : Path
        Path to the dataset directory.

    Returns
    -------
    bool
        True if the dataset is valid, False otherwise.
    """
    if not dataset_path.exists():
        print("Dataset path does not exist.")
        return False

    folder = ["train", "valid", "test"]
    csvs = ["annotated_train.csv", "annotated_valid.csv", "annotated_test.csv"]

    # Check if required subfolders and CSV files exist
    for i in folder + csvs:
        if not (dataset_path / i).exists():
            print(f"'{i}' file or folder does not exist.")
            return False

    # Validate CSV content
    for i in csvs:
        data = pd.read_csv(dataset_path / i)

        # Validate columns
        if list(data.columns) != ["img_name", "img_path", "class_idx", "class_name"]:
            print(
                f"'{i}' file does not have the correct columns.\n"
                f"Expected: ['img_name', 'img_path', 'class_idx', 'class_name']\n"
            )
            return False

        # Validate images and class indices
        for _, row in data.iterrows():
            if not (dataset_path / row["img_path"]).exists():
                print(f"Image '{row['img_path']}' in '{i}' file does not exist.")
                return False

            if not isinstance(row["class_idx"], int):
                print(f"Class index in '{i}' file is not an integer.")
                return False

    return True


def find_folders_with_model(folder: str) -> List[Path]:
    """
     searches for folders containing a model file named 'model.pth'.

    Parameters
    ----------
    folder : str
        Root directory to start searching from.

    Returns
    -------
    List[Path]
        A list of paths to folders containing a `model.pth` file.
    """
    matching_folders = []
    for root, dirs, files in os.walk(folder):
        if "model.pth" in files:
            matching_folders.append(Path(root))
    return matching_folders


def save_json(data: Union[Dict, List], filepath: Path, default: Any = None):
    """
    Saves JSON data to a file.

    Parameters
    ----------
    data : Union[Dict, List]
        JSON-compatible data (dictionary or list) to be saved.

    filepath : Path
        Path where the JSON file will be written.

    default : Any, optional
        A function used to handle objects that are not JSON serializable.
        Defaults to None.

    Raises
    ------
    TypeError
        If the data provided is not serializable to JSON.
    IOError
        If there is an issue writing the file to the specified path.
    """
    with filepath.open(mode="w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=default)


def load_json(filepath: Path) -> Union[Dict, List]:
    """
    Loads JSON data from a file.

    Parameters
    ----------
    filepath : Path
        Path of the JSON file to load.

    Returns
    -------
    Union[Dict, List]
        The parsed JSON content.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    json.JSONDecodeError
        If the file contains invalid JSON.
    IOError
        If there is an issue reading the file.
    """
    with filepath.open(mode="r", encoding="utf-8") as file:
        dict_ = json.load(file)
    return dict_