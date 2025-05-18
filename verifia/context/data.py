import logging
from typing import Iterable, List, Optional, Union, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Dataset:
    """
    Represents a dataset containing features and a target label. Provides functionality for
    sampling, and supports filtering based on domain-specific criteria.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_name: str,
        feature_names: Optional[Iterable[str]] = None,
        cat_feature_names: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Initializes the Dataset instance by validating the input DataFrame and inferring
        categorical features if not provided.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data.
            target_name (str): Name of the target (label) column.
            feature_names (Optional[Iterable[str]]): Iterable of all feature names.
                If None, all features names are automatically inferred from columns in
                the provided dataframe.
            cat_feature_names (Optional[Iterable[str]]): Iterable of categorical feature names.
                If None, categorical features are automatically inferred from columns in
                feature_names with dtype 'object'.

        Raises:
            ValueError: If the DataFrame does not contain all required columns.
        """
        self._target_name: str = target_name
        if feature_names is None:
            self._feature_names: List[str] = [
                col for col in df.columns if col != target_name
            ]
        else:
            self._feature_names = list(feature_names)
        self._feature_names: List[str] = list(feature_names)

        # Verify that required columns are present
        required_columns = self._feature_names + [self._target_name]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"DataFrame is missing columns: {missing_columns}")

        # Create a copy of the DataFrame containing only the required columns
        self._data: pd.DataFrame = df[required_columns].copy()

        # Infer categorical feature names if not provided
        if cat_feature_names is None:
            self._cat_feature_names: List[str] = [
                col for col in self._feature_names if self._data[col].dtype == object
            ]
        else:
            self._cat_feature_names = list(cat_feature_names)

    def feature_data(self, enforced_data_type=False) -> pd.DataFrame:
        """
        Retrieve the features subset from the dataset.

        Returns:
            pd.DataFrame: A DataFrame containing only the columns specified in self._feature_names.
        """
        if enforced_data_type:
            feature_df = self._data[self._feature_names].copy()
            for num_feat in self.num_feature_names:
                feature_df[num_feat] = feature_df[num_feat].astype("float")
            for cat_feat in self.cat_feature_names:
                feature_df[cat_feat] = feature_df[cat_feat].astype("category")
            return feature_df
        else:
            return self._data[self._feature_names].copy()

    @property
    def target_data(self) -> pd.Series:
        """
        Retrieve the target column from the dataset.

        Returns:
            pd.Series: A Series corresponding to the target variable specified by self._target_name.
        """
        return self._data[self._target_name].copy()

    @property
    def X(self) -> npt.NDArray:
        """
        Returns:
            npt.NDArray: NumPy array containing the feature data.
        """
        return self._data[self._feature_names].to_numpy()

    @property
    def y(self) -> npt.NDArray:
        """
        Returns:
            npt.NDArray: NumPy array containing the target (label) data.
        """
        return self._data[self._target_name].to_numpy()

    @property
    def num_feature_names(self):
        """
        Return the list of numerical feature names in their original order.

        Returns:
            List[str]: A list of feature names that are not categorical.
        """
        return list(set(self._feature_names) - set(self.cat_feature_names))

    @property
    def cat_feature_names(self):
        """
        Return the list of categorical feature names.

        Returns:
            List[str]: A list of categorical feature names.
        """
        return self._cat_feature_names

    @property
    def num_feature_idxs(self):
        """
        Return the list of indices corresponding to numeric (non-categorical) features.

        Returns:
            List[int]: A list of indices in self._feature_names for numeric features.
        """
        cat_set = set(self.cat_feature_names)
        return [
            idx for idx, col in enumerate(self._feature_names) if col not in cat_set
        ]

    @property
    def cat_feature_idxs(self):
        """
        Return the list of indices corresponding to categorical features.

        Returns:
            List[int]: A list of indices in self._feature_names for categorical features.
        """
        cat_set = set(self.cat_feature_names)
        return [idx for idx, col in enumerate(self._feature_names) if col in cat_set]

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: A copy of the internal DataFrame containing the dataset.
        """
        return self._data.copy()

    def split(
        self, primary_split_size: float, random_state: Optional[int] = None
    ) -> Tuple["Dataset", "Dataset"]:
        """
        Split the dataset into a primary split and a secondary split.

        Args:
            primary_split_size (float): The fraction of the dataset to include in the primary split.
            random_state (Optional[int], optional): Seed for reproducibility. Defaults to None.

        Returns:
            Tuple[Dataset, Dataset]: A tuple containing:
                - A Dataset for the primary split.
                - A Dataset for the secondary split.
        """
        primary_split_df, secondary_split_df = train_test_split(
            self._data.copy(), train_size=primary_split_size, random_state=random_state
        )
        return (
            Dataset(
                primary_split_df,
                self._target_name,
                self._feature_names,
                self._cat_feature_names,
            ),
            Dataset(
                secondary_split_df,
                self._target_name,
                self._feature_names,
                self._cat_feature_names,
            ),
        )

    def sample(
        self,
        n_samples: Optional[int] = None,
        prop_samples: Optional[float] = None,
        replace: bool = False,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> "Dataset":
        """
        Creates a new Dataset instance by sampling rows from the current dataset.

        Either `n_samples` or `prop_samples` must be provided. If both are provided,
        `n_samples` takes precedence.

        Args:
            n_samples (Optional[int]): The exact number of samples to extract. Must be positive.
            prop_samples (Optional[float]): Proportion of the total samples to extract (between 0 and 1).
            replace (bool): Whether sampling is done with replacement. Defaults to False.
            random_state (Optional[Union[int, np.random.Generator]]): Seed or Generator for reproducible sampling.

        Returns:
            Dataset: A new Dataset instance containing the sampled data.

        Raises:
            ValueError: If neither `n_samples` nor `prop_samples` is provided, if provided values are out of range,
                        or if sampling without replacement and `n_samples` exceeds available data.
        """
        if n_samples is None and prop_samples is None:
            raise ValueError("You must provide either n_samples or prop_samples.")

        if n_samples is not None:
            if n_samples <= 0:
                raise ValueError("n_samples must be a positive integer.")
        else:
            if not (0.0 <= prop_samples <= 1.0):
                raise ValueError("prop_samples must be within [0.0, 1.0].")
            n_samples = int(len(self._data) * prop_samples)

        if not replace and n_samples > len(self._data):
            raise ValueError(
                "n_samples cannot be greater than the number of data points when sampling without replacement."
            )

        # Initialize random number generator
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(self._data), size=n_samples, replace=replace)
        sampled_df = self._data.iloc[indices].copy()
        return Dataset(
            sampled_df, self._target_name, self._feature_names, self._cat_feature_names
        )

    @property
    def n_samples(self) -> int:
        """
        Returns:
            int: The number of samples in the dataset.
        """
        return self._data.shape[0]

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples in the dataset.
        """
        return self.n_samples
