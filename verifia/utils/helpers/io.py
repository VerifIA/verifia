import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Union

import json
import yaml
import shutil
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def find_pdf_file_paths(pdf_folder_path: str) -> List[str]:
    """
    Recursively traverse the given folder and return a list of paths to all PDF files.

    Args:
        pdf_folder_path (str): The path to the directory where PDF files are located.

    Returns:
        List[str]: A list of file paths to PDF files found within the directory and its subdirectories.
    """
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(pdf_folder_path)
        for file in files
        if file.lower().endswith(".pdf")
    ]


def read_config_file(fpath: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Read a configuration file into a dictionary. Supports JSON and YAML formats.

    Parameters:
        fpath (Union[str, os.PathLike]): The file path to the configuration file.

    Returns:
        Dict[str, Any]: Dictionary representation of the configuration data.

    Raises:
        ValueError: If the file extension is not supported.
        Exception: For any errors encountered during file reading.
    """
    try:
        fpath_str = str(fpath)
    except Exception as e:
        raise ValueError("The provided file path is not valid.") from e

    ext = os.path.splitext(fpath_str)[1].lower()

    if ext == ".json":
        return read_json(fpath_str)
    elif ext in [".yaml", ".yml"]:
        return read_yaml(fpath_str)
    else:
        raise ValueError(
            f"Unsupported config file extension '{ext}'. Supported formats are: JSON, YAML."
        )


def validate_dirpath(dirpath: Union[str, os.PathLike]) -> None:
    """
    Validate that the given path is a directory.

    Args:
        dirpath (Union[str, os.PathLike]): The path to validate.

    Raises:
        ValueError: If the provided path is not a valid directory.
    """
    if not os.path.isdir(str(dirpath)):
        raise ValueError(f"Provided path '{dirpath}' is not a valid directory.")


def validate_filepath(local_fpath: Union[str, os.PathLike]) -> None:
    """
    Validate that the given path is a file and is not empty.

    Args:
        local_fpath (Union[str, os.PathLike]): The file path to validate.

    Raises:
        ValueError: If the file does not exist or is empty.
    """
    local_fpath_str = str(local_fpath)
    if not os.path.isfile(local_fpath_str):
        raise ValueError(f"Provided path '{local_fpath}' is not a valid file.")
    if os.path.getsize(local_fpath_str) == 0:
        raise ValueError(f"Provided file '{local_fpath}' is empty.")


def read_yaml(local_path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Read a YAML file and return its contents as a dictionary.

    Args:
        local_path (Union[str, os.PathLike]): The file path to the YAML file.

    Returns:
        Dict[str, Any]: The parsed YAML data as a dictionary.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist.
        yaml.YAMLError: If there is an error during YAML parsing.
    """
    meta_file = Path(local_path)
    if not meta_file.exists():
        raise FileNotFoundError(f"{local_path} does not exist.")

    with meta_file.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data


def read_json(local_path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Read a JSON file and return its contents as a dictionary.

    Args:
        local_path (Union[str, os.PathLike]): The file path to the JSON file.

    Returns:
        Dict[str, Any]: The parsed JSON data as a dictionary.

    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
        json.JSONDecodeError: If the JSON file contains invalid JSON.
    """
    meta_file = Path(local_path)
    if not meta_file.exists():
        raise FileNotFoundError(f"{local_path} does not exist.")

    with meta_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def save_yaml(yaml_fpath: Union[str, os.PathLike], data: Dict[str, Any]) -> None:
    """
    Save a dictionary to a YAML file.

    This function writes a dictionary to a specified YAML file. If the parent directory
    does not exist, it will be created automatically.

    Args:
        yaml_fpath (Union[str, os.PathLike]): The path where the YAML file will be saved.
        data (Dict[str, Any]): The dictionary data to be written.

    Raises:
        OSError: If there is an issue with writing to the file system.
        yaml.YAMLError: If an error occurs during YAML serialization.
    """
    yaml_fpath = Path(yaml_fpath)
    yaml_fpath.parent.mkdir(parents=True, exist_ok=True)
    with yaml_fpath.open("w", encoding="utf-8") as f:
        # yaml.dump(data, f)
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(json_fpath: Union[str, os.PathLike], data: Dict[str, Any]) -> None:
    """
    Save a dictionary to a JSON file, handling numpy data types.

    This function serializes a dictionary to a JSON file, converting numpy arrays
    and numpy-specific data types (e.g., np.float32, np.int64) to standard Python types.

    Args:
        json_fpath (Union[str, os.PathLike]): The path where the JSON file will be saved.
        data (Dict[str, Any]): The dictionary data to be written.

    Raises:
        OSError: If there is an issue with writing to the file system.
        json.JSONDecodeError: If an error occurs during JSON serialization.
    """

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return super().default(obj)

    json_fpath = Path(json_fpath)
    json_fpath.parent.mkdir(parents=True, exist_ok=True)
    with json_fpath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)


def first_file_dot_ext_in_folder(
    folder_path: Union[str, os.PathLike], ext: str
) -> Path:
    """
    Recursively search for a file with the given extension in a folder and its sub-folders.
    Raises an exception if multiple or no matching files are found.

    Parameters:
        folder_path (Union[str, os.PathLike]): The path to the folder.
        ext (str): The file extension to search for (e.g., '.txt').

    Returns:
        Path: The full path to the file with the specified extension.

    Raises:
        Exception: If no file or multiple files with the extension are found.
    """
    if not ext.startswith("."):
        ext = "." + ext

    matching_files: List[str] = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(ext):
                matching_files.append(os.path.join(root, file))

    if not matching_files:
        raise Exception(
            f"No files with extension '{ext}' found in folder '{folder_path}' or its sub-folders."
        )
    elif len(matching_files) > 1:
        raise Exception(
            f"Multiple files with extension '{ext}' found: {matching_files}"
        )

    return Path(matching_files[0])


def mk_tmpdir(tmp_dir_path: Union[str, os.PathLike] = "tmp") -> Path:
    """
    Create a temporary directory if it does not already exist.

    Args:
        tmp_dir_path (Union[str, os.PathLike], optional): The path of the temporary folder.
            Defaults to 'tmp'.

    Returns:
        Path: The path to the temporary directory.
    """
    tmp_dir_path = Path(tmp_dir_path)
    if not tmp_dir_path.exists():
        tmp_dir_path.mkdir(parents=True, exist_ok=True)
    return tmp_dir_path


def rm_tmpdir(tmp_dir_path: Union[str, os.PathLike] = "tmp") -> None:
    """
    Remove the specified temporary directory and all its contents.

    Args:
        tmp_dir_path (Union[str, os.PathLike], optional): The path of the temporary folder.
            Defaults to 'tmp'.

    Raises:
        OSError: If an error occurs while removing the directory.
    """
    tmp_dir_path = Path(tmp_dir_path)
    if tmp_dir_path.exists():
        shutil.rmtree(tmp_dir_path)
