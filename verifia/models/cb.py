import os
import logging
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import numpy.typing as npt

from .base import BaseModel, ModelOutputs
from ..utils.helpers.optional import require
from ..utils.enums.models import SupportedModelTypes, SupportedMLFrameworks

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class CBModel(BaseModel):
    """
    A CatBoost model wrapper that extends the BaseModel interface.

    Provides methods to save, load, and make predictions using CatBoost models.
    Depending on the model type (classification or regression), the appropriate
    CatBoost estimator is used.
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
        Initialize a CatBoost model wrapper.

        Args:
            name (str): The model name.
            version (str): The model version.
            model_type (SupportedModelTypes): Type of the model (classification/regression).
            feature_names (Iterable): Iterable of feature names.
            target_name (str): Target variable name.
            cat_feature_names (Optional[Iterable]): Optional iterable of categorical feature names.
            classification_threshold (Optional[float]): Threshold for classification decisions.
            description (Optional[str]): Optional model description.
        """
        super().__init__(
            model_type=model_type,
            feature_names=feature_names,
            target_name=target_name,
            local_dirpath=local_dirpath,
            cat_feature_names=cat_feature_names,
            classification_threshold=classification_threshold,
            name=name,
            version=version,
            description=description,
            framework=SupportedMLFrameworks.CB,
        )
        # The model instance will be set when load_model is called.
        self._catboost = require("catboost", "catboost")
        self.model = None

    def save_model(self, path: os.PathLike = None, *args, **kwargs) -> None:
        """
        Save the CatBoost model to the specified file path.

        Args:
            path (os.PathLike): The file path where the model should be saved.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the model attribute is not set.
            Exception: Propagates any exception raised by the underlying save_model method.
        """
        if path is None:
            local_path = self.default_model_filepath()
        else:
            local_path = Path(path)

        local_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise ValueError("No model to save. Load or train a model first.")
        try:
            self.model.save_model(str(local_path), *args, **kwargs)
            logger.info("Model saved successfully to %s", local_path)
        except Exception:
            logger.exception("Failed to save model to %s", local_path)
            raise

    def load_model(self, path: os.PathLike = None, *args, **kwargs) -> "BaseModel":
        """
        Load the CatBoost model from the specified file path.

        If the provided path is a directory, the default model file path
        (constructed by `default_model_filepath`) is used.

        Args:
            path (os.PathLike): Path (or directory) from which to load the model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            BaseModel: Self, to allow method chaining.

        Raises:
            ValueError: If the model type is unsupported.
            Exception: Propagates any exception raised by the underlying load_model method.
        """
        local_path = self.default_model_filepath() if path is None else Path(path)
        try:
            if self.model_type == SupportedModelTypes.CLASSIFICATION:
                self.model = self._catboost.CatBoostClassifier()
            elif self.model_type == SupportedModelTypes.REGRESSION:
                self.model = self._catboost.CatBoostRegressor()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            self.model.load_model(str(local_path), *args, **kwargs)
            logger.info("Model loaded successfully from %s", local_path)
            return self
        except Exception:
            logger.exception("Failed to load model from %s", local_path)
            raise

    def predict_score(self, data: npt.NDArray) -> Union[npt.NDArray, float]:
        """
        Compute prediction scores for the given data.

        For regression, the raw predictions are returned.
        For classification, the probability of the positive class is returned.

        Args:
            data (npt.NDArray): The input data as a NumPy array. Can be 1D or 2D.

        Returns:
            Union[npt.NDArray, float]: A NumPy array of prediction scores, or a single score if one sample is provided.

        Raises:
            Exception: Propagates any exception raised during prediction.
        """
        one_entry = data.ndim == 1
        data = data.reshape(1, -1) if one_entry else data
        try:
            if self.is_regression:
                scores = self.model.predict(data)
            else:
                scores = self.model.predict_proba(data)[:, 1]
            return scores[0] if one_entry else scores
        except Exception:
            logger.exception("Error during prediction scoring.")
            raise

    def predict(self, data: npt.NDArray) -> ModelOutputs:
        """
        Generate predictions and, if applicable, probabilities for the given data.

        For regression, only predictions are populated.
        For classification, both predictions and probabilities are populated.

        Args:
            data (npt.NDArray): The input data as a NumPy array. Can be 1D or 2D.

        Returns:
            ModelOutputs: An instance containing predictions and probabilities (if classification).

        Raises:
            Exception: Propagates any exception raised during prediction.
        """
        one_entry = data.ndim == 1
        data = data.reshape(1, -1) if one_entry else data

        outputs = ModelOutputs()
        try:
            if self.is_regression:
                outputs.predictions = self.model.predict(data)
            else:
                outputs.probabilities = self.model.predict_proba(data)[:, 1]
                # Compute binary predictions based on the classification threshold.
                outputs.predictions = (
                    outputs.probabilities > self.classification_threshold
                ).astype(np.int32)
            return outputs
        except Exception:
            logger.exception("Error during prediction.")
            raise
