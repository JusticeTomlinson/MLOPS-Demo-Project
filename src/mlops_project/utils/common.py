import os
import box
from box.exceptions import BoxValueError
import yaml
from src.mlops_project import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations

def save_json(path, data):
    import json
    with open(path, 'w') as f:
        # Convert NumPy types to Python native types explicitly
        converted_data = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in data.items()}
        json.dump(converted_data, f)

# def save_json(path: Path, data: dict):
#     """save json data

#     Args:
#         path (Path): path to json file
#         data (dict): data to be saved in json file
#     """
#     with open(path, "w") as f:
#         json.dump(data, f, indent=4)

#     logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"



def resample_classes(dataframe):
    df_majority = dataframe[dataframe["CHDRisk"] == 0]
    df_minority = dataframe[dataframe["CHDRisk"] == 1]

    minority_count = len(df_minority)

    df_majority_downsampled = df_majority.sample(n=minority_count, random_state=42)

    df_balanced = pd.concat([df_minority, df_majority_downsampled])

    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    #Turn all items within the table to float
    df_balanced = df_balanced.astype(float)
    return df_balanced