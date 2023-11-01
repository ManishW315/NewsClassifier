import os

import pandas as pd
import yaml

from newsclassifier.config.config import Cfg, logger


def write_yaml(data: pd.DataFrame, filepath: str):
    logger.info("Writing yaml file.")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def read_yaml(file_path: str):
    logger.info("Reading yamlfile")
    with open(file_path, "r") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return params
