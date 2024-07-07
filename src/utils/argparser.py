import os
import yaml
from typing import Dict, Union

def get_args(args_path: str) -> Dict[str, Union[float, str]]:
    """
    Gets relevant arguments from a yaml file.

    Parameters
    ----------
    args_path: str
        The path to the yaml file containing the arguments.
    
    Returns
    -------    
    args: Dict[str, Union[float, str]]
        The arguments in the form of a dictionary.
    """
    
    with open(args_path, "r") as f:
        args = yaml.safe_load(f)

    return args

def save_args(log_dir: str, args: Dict[str, Union[float, str]]) -> None:
    """
    Saves arguments inside a log directory.

    log_dir: str
        The destination directory to save the arguments to.

    args: Dict[str, Union[float, str]]
        The arguments to be saved. The resulting yaml file will have a filename `run_config.yaml`.
    """

    path = os.path.join(log_dir, "run_config.yaml")
    
    with open(path, "w") as f:
        yaml.dump(args, f)
