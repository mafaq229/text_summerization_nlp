import os
import yaml
from pathlib import Path
from typing import Any

from ensure import ensure_annotations
from box.exceptions import BoxValueError
from box import ConfigBox

from src.textsummarizer.logging import logger


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
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content) # config box allows us to access key, value pair info as property i.e "dict.key" will "value"
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
 
@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for dir in path_to_directories:
        os.makedirs(dir, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {dir}")
