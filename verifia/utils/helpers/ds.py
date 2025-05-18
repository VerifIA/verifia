import os
from collections.abc import Iterable
from typing import List, Any, Dict

import tqdm
import pandas as pd
import numpy as np
import numpy.typing as npt


def remove_duplicates(
    dict_list: List[Dict[str, Any]], keys: List[str], tol: float = 1e-6
) -> List[Dict[str, Any]]:
    """
    Remove duplicate dictionaries from a list based on a subset of keys.
    For keys with floating point values, values are considered equal if they
    are within the specified tolerance.

    Parameters:
        dict_list (List[Dict[str, Any]]): List of dictionaries to filter.
        keys (List[str]): Keys used for determining duplicates.
        tol (float): Tolerance for floating point comparisons.

    Returns:
        List[Dict[str, Any]]: A new list with duplicates removed.
    """
    seen = set()
    unique_dicts = []
    # Determine number of decimals based on the tolerance.
    decimals = int(-np.log10(tol))

    def normalize(val: Any) -> Any:
        # If the value is a float, round it to the specified number of decimals.
        if isinstance(val, float):
            return round(val, decimals)
        return val

    for d in dict_list:
        # Create a tuple identifier using the normalized value for each specified key.
        identifier = tuple(normalize(d.get(key)) for key in keys)
        if identifier not in seen:
            seen.add(identifier)
            unique_dicts.append(d)
    return unique_dicts


def feat_vect2dict(feature_names: List[str], feat_vect: npt.NDArray) -> Dict[str, Any]:
    """
    Convert a feature vector to a dictionary mapping feature names to values.

    Args:
        feature_names (List[str]): The list of feature names.
        feat_vect (npt.NDArray): The feature vector.

    Returns:
        Dict[str, Any]: A dictionary mapping each feature name to its corresponding value.

    Raises:
        ValueError: If the length of the feature vector does not match the number of feature names.
    """
    if len(feature_names) != len(feat_vect):
        raise ValueError(
            "Length of feature vector does not match the number of feature names."
        )
    return dict(zip(feature_names, feat_vect))


def read_data_file(fpath: os.PathLike, **kwargs) -> pd.DataFrame:
    """
    Read a data file into a pandas DataFrame. Supported formats include CSV, Excel,
    JSON, Parquet, Feather, and Pickle.

    Parameters:
        fpath (os.PathLike): The file path to the data.
        **kwargs: Additional keyword arguments to pass to the corresponding Pandas read function.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.

    Raises:
        ValueError: If the file extension is not supported.
        Exception: For any errors encountered during file reading.
    """
    try:
        fpath = str(fpath)
    except Exception as e:
        raise ValueError("The provided file path is not valid.") from e

    ext = os.path.splitext(fpath)[1].lower()

    if ext == ".csv":
        return pd.read_csv(fpath, **kwargs)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(fpath, **kwargs)
    elif ext == ".json":
        return pd.read_json(fpath, **kwargs)
    elif ext == ".parquet":
        return pd.read_parquet(fpath, **kwargs)
    elif ext == ".feather":
        return pd.read_feather(fpath, **kwargs)
    elif ext == ".pkl":
        return pd.read_pickle(fpath, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported formats are: CSV, Excel, JSON, Parquet, Feather, and Pickle."
        )


def create_progress_bar(iterable: Iterable[Any], desc: str) -> tqdm.tqdm:
    """
    Wrap an iterable with a tqdm progress bar.

    Args:
        iterable (Iterable[Any]): The iterable to wrap.
        desc (str): Description for the progress bar.

    Returns:
        tqdm.tqdm: A progress bar iterator.
    """
    return tqdm.tqdm(iterable, desc=desc)


def reorder_dataframe_columns(
    data: npt.NDArray,
    current_feature_names: List[str],
    desired_feature_names: List[str],
) -> npt.NDArray:
    """
    Reorganize the given data array to match a new order of feature names.

    Args:
        data (npt.NDArray): Input data array of shape (n_rows, n_features).
        current_feature_names (List[str]): The current order of feature names.
        desired_feature_names (List[str]): The desired new order of feature names.

    Returns:
        npt.NDArray: The data array reorganized to match the desired feature order.

    Raises:
        ValueError: If the set of desired feature names does not exactly match the current feature names.
    """
    if set(current_feature_names) != set(desired_feature_names):
        raise ValueError(
            "The new feature names must match the current feature names exactly."
        )

    current_index_map = {name: idx for idx, name in enumerate(current_feature_names)}
    new_index_order = [current_index_map[name] for name in desired_feature_names]
    reordered = data[:, new_index_order]
    return reordered


def get_data_types_by_column(array: npt.NDArray) -> List[type]:
    """
    Determine the type of the first element in each column of a NumPy array.

    Args:
        array (npt.NDArray): The input array.

    Returns:
        List[type]: A list where each element is the type of the first element of each column.
    """
    types_by_column = []
    for col in range(array.shape[1]):
        types_by_column.append(type(array[0, col]))
    return types_by_column
