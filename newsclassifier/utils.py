import os
from typing import Dict

import pandas as pd
import yaml

from newsclassifier.config.config import logger


def write_yaml(data: pd.DataFrame, filepath: str) -> None:
    """Write into YAML file.

    Args:
        data (pd.DataFrame): Data to be parsed into YAML
        filepath (str): Path of location to be stored at.
    """
    logger.info("Writing yaml file.")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def read_yaml(file_path: str) -> Dict:
    """Read YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        Dict: Read parameters.
    """
    logger.info("Reading yamlfile")
    with open(file_path, "r") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return params
