import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import logging

from .base import BaseModel, ModelOutputs
from ..utils.enums.models import SupportedModelTypes, SupportedMLFrameworks
from ..utils.helpers.ds import get_data_types_by_column
from ..utils.helpers.optional import require

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LGBModel(BaseModel):
    """
    A LightGBM model wrapper that extends BaseModel for regression and classification tasks.

    This class provides methods to save, load, and make predictions using LightGBM models.
    """

    def __init__(
        self,
        name: str,
        version: str,
        model_type: SupportedModelTypes,
        feature_names: Iterable,
        target_name: str,
        local_dirpath: str,
        cat_feature_names: Optional[Iterable] = None,
        classification_threshold: Optional[float] = 0.5,
        description: Optional[str] = None,
    ) -> None:
        """
        Initialize a LightGBM model wrapper.

        Args:
            name (str): The model name.
            version (str): The model version.
            model_type (SupportedModelTypes): Type of the model (classification or regression).
            feature_names (Iterable): Iterable of feature names.
            target_name (str): The target variable name.
            cat_feature_names (Optional[Iterable]): Optional iterable of categorical feature names.
            classification_threshold (Optional[float]): Threshold for classification tasks.
            description (Optional[str]): Optional model description.
        """
        super().__init__(
            name=name,
            version=version,
            model_type=model_type,
            feature_names=feature_names,
            target_name=target_name,
            local_dirpath=local_dirpath,
            cat_feature_names=cat_feature_names,
            classification_threshold=classification_threshold,
            description=description,
            framework=SupportedMLFrameworks.LGBM,
        )
        self.model = None

    def save_model(self, path: os.PathLike = None, *args, **kwargs) -> None:
        """
        Save the LightGBM model to the specified file path.

        Args:
            path (os.PathLike): The file path (or directory) where the model should be saved.
            *args: Additional positional arguments for LightGBM's save_model.
            **kwargs: Additional keyword arguments for LightGBM's save_model.

        Raises:
            ValueError: If no model instance is available.
            Exception: Propagates exceptions raised during model saving.
        """
        if path is None:
            local_path = self.default_model_filepath()
        else:
            local_path = Path(path)

        local_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise ValueError(
                "No model instance available to save. Load or train the model first."
            )
        try:
            self.model.save_model(str(local_path), *args, **kwargs)
            logger.info("LightGBM model saved successfully to %s", local_path)
        except Exception:
            logger.exception("Failed to save LightGBM model to %s", local_path)
            raise

    def load_model(self, path: os.PathLike = None, *args, **kwargs) -> "BaseModel":
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "To use LGBModel.load_model() you must install lightgbm: "
                "`pip install verifia[lightgbm]`"
            ) from e
        """
        Load a LightGBM model from the specified file path.

        If the provided path is a directory, the default model file path is used.

        Args:
            path (os.PathLike): The file path (or directory) from which to load the model.
            *args: Additional positional arguments for LightGBM's load_model.
            **kwargs: Additional keyword arguments for LightGBM's load_model.

        Returns:
            BaseModel: Self, to allow method chaining.

        Raises:
            Exception: Propagates exceptions raised during model loading.
        """
        local_path = self.default_model_filepath() if path is None else Path(path)
        logger.info("Loading LightGBM model from %s", local_path)
        try:
            self.model = lgb.Booster(model_file=str(local_path))
            logger.info("LightGBM model loaded successfully from %s", local_path)
            return self
        except Exception:
            logger.exception("Failed to load LightGBM model from %s", local_path)
            raise

    def prepare_inputs(self, X: npt.NDArray) -> pd.DataFrame:
        """
        Prepare and convert the input NumPy array into a Pandas DataFrame with proper dtypes.

        Args:
            X (npt.NDArray): Input feature data as a NumPy array.

        Returns:
            pd.DataFrame: DataFrame with column names matching the model's features and appropriate types.
        """
        # Infer data types for each column using a helper function.
        feature_types = get_data_types_by_column(X)
        # Use model's feature names if available; otherwise, fallback to the model card's feature names.
        # feature_names = self.model.feature_name() if self.model is not None else self.feature_names
        proc_data = pd.DataFrame(X, columns=self.feature_names)
        for fname, ftype in zip(self.feature_names, feature_types):
            # For string types, cast to 'category'; otherwise, use the inferred type.
            proc_data[fname] = proc_data[fname].astype(
                "category" if ftype == str else ftype
            )
        return proc_data

    def predict_score(self, data: npt.NDArray) -> npt.NDArray:
        """
        Compute prediction scores for the input data.

        For regression, returns raw predictions.
        For classification, returns the probability of the positive class.

        Args:
            data (npt.NDArray): Input feature data as a NumPy array (1D or 2D).

        Returns:
            npt.NDArray: A NumPy array of prediction scores.
        """
        one_entry = data.ndim == 1
        data = data.reshape(1, -1) if one_entry else data
        data = self.prepare_inputs(data)
        try:
            scores = self.model.predict(data)
            return scores[0] if one_entry else scores
        except Exception:
            logger.exception("Error during prediction scoring.")
            raise

    def predict(self, data: npt.NDArray) -> ModelOutputs:
        """
        Generate predictions for the input data.

        For regression tasks, only predictions are produced.
        For classification tasks, both predictions and probabilities are produced.

        Args:
            data (npt.NDArray): Input feature data as a NumPy array (1D or 2D).

        Returns:
            ModelOutputs: An object containing predictions and probabilities (if applicable).
        """
        one_entry = data.ndim == 1
        data = data.reshape(1, -1) if one_entry else data
        data = self.prepare_inputs(data)
        pred_outs = ModelOutputs()
        try:
            if self.is_regression:
                pred_outs.predictions = self.model.predict(data)
            else:
                probabilities = self.model.predict(data)
                pred_outs.probabilities = probabilities
                pred_outs.predictions = np.int32(
                    probabilities > self.classification_threshold
                )
            return pred_outs
        except Exception:
            logger.exception("Error during prediction.")
            raise
