import os
import logging
from typing import Iterable, Optional
from pathlib import Path

import numpy.typing as npt
import pandas as pd
from .base import BaseModel, ModelOutputs
from ..utils.enums.models import SupportedModelTypes, SupportedMLFrameworks
from ..utils.helpers.optional import require

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class XGBModel(BaseModel):
    """
    XGBoost model wrapper that implements the BaseModel interface for both
    regression and classification tasks.
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
        Initialize the XGBModel wrapper.

        Args:
            name (str): The model name.
            version (str): The model version.
            model_type (SupportedModelTypes): Type of the model (e.g., classification or regression).
            feature_names (Iterable): Iterable of feature names.
            target_name (str): The target variable name.
            cat_feature_names (Optional[Iterable]): Optional iterable of categorical feature names.
            classification_threshold (Optional[float]): Classification threshold (for classification tasks).
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
            framework=SupportedMLFrameworks.XGB,
        )
        self.model = None
        self._xgboost = require("xgboost", "xgboost")

    def save_model(self, path: os.PathLike = None, *args, **kwargs) -> None:
        """
        Save the XGBoost model to the specified file path.

        Args:
            path (os.PathLike): The file path where the model should be saved.
            *args: Additional positional arguments for the XGBoost save_model method.
            **kwargs: Additional keyword arguments for the XGBoost save_model method.

        Raises:
            Exception: Propagates exceptions raised during model saving.
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
            logger.info("XGBoost model saved successfully to %s", local_path)
        except Exception:
            logger.exception("Failed to save XGBoost model to %s", local_path)
            raise

    def load_model(self, path: os.PathLike = None, *args, **kwargs) -> "BaseModel":
        """
        Load an XGBoost model from the specified file path.

        If the provided path is a directory, the default model filepath is used.

        Args:
            path (os.PathLike): The file path (or directory) from which to load the model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            BaseModel: Self, to allow method chaining.

        Raises:
            Exception: Propagates exceptions raised during model loading.
        """
        local_path = self.default_model_filepath() if path is None else Path(path)
        try:
            if self.model_type == SupportedModelTypes.CLASSIFICATION:
                self.model = self._xgboost.XGBClassifier()
            elif self.model_type == SupportedModelTypes.REGRESSION:
                self.model = self._xgboost.XGBRegressor()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            self.model.load_model(str(local_path))
            logger.info("XGBoost model loaded successfully from %s", local_path)
            return self
        except Exception:
            logger.exception("Failed to load XGBoost model from %s", local_path)
            raise

    def prepare_inputs(self, X: npt.NDArray) -> pd.DataFrame:
        """
        Prepare input data by converting the NumPy array to a DataFrame with appropriate dtypes.

        This method assumes that the model's feature types are stored in its
        'feature_types' attribute, where a type value of "c" indicates a categorical feature.

        Args:
            X (npt.NDArray): Input feature data.

        Returns:
            pd.DataFrame: Processed data with columns renamed and dtypes set.

        Raises:
            Exception: Propagates exceptions raised during data processing.
        """
        try:
            proc_data = pd.DataFrame(X, columns=self.feature_names)
            # Convert each column to the correct dtype based on the model's feature_types.
            for fname, ftype in zip(self.feature_names, self.model.feature_types):
                proc_data[fname] = proc_data[fname].astype(
                    "category" if ftype == "c" else ftype
                )
            return proc_data
        except Exception:
            logger.exception("Error during input preparation.")
            raise

    def predict_score(self, data: npt.NDArray) -> npt.NDArray:
        """
        Compute prediction scores for the given input data.

        For regression models, returns raw predictions.
        For classification models, returns the probability of the positive class.

        Args:
            data (npt.NDArray): Input data array (1D or 2D).

        Returns:
            npt.NDArray: Array of prediction scores.

        Raises:
            Exception: Propagates exceptions raised during prediction.
        """
        one_entry = data.ndim == 1
        data = data.reshape(1, -1) if one_entry else data
        try:
            data = self.prepare_inputs(data)
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
        Generate predictions for the given input data.

        For regression tasks, only predictions are produced.
        For classification tasks, produces predictions, probabilities, and maps predictions to labels.

        Args:
            data (npt.NDArray): Input data array (1D or 2D).

        Returns:
            ModelOutputs: An object containing predictions and, for classification, probabilities.

        Raises:
            Exception: Propagates exceptions raised during prediction.
        """
        one_entry = data.ndim == 1
        data = data.reshape(1, -1) if one_entry else data
        pred_outs = ModelOutputs()
        try:
            data = self.prepare_inputs(data)
            if self.is_regression:
                pred_outs.predictions = self.model.predict(data)
            else:
                pred_outs.predictions = self.model.predict(data)
                pred_outs.probabilities = self.model.predict_proba(data)[:, 1]
            return pred_outs
        except Exception:
            logger.exception("Error during prediction.")
            raise
