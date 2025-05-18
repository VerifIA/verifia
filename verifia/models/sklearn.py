import os
import logging
from typing import Iterable, Optional
from pathlib import Path
import numpy.typing as npt
import cloudpickle
from .base import BaseModel, ModelOutputs
from ..utils.enums.models import SupportedModelTypes, SupportedMLFrameworks

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SKLearnModel(BaseModel):
    """
    A scikit-learn model wrapper that implements the BaseModel interface.

    This class provides methods to save, load, and perform predictions using models
    trained with scikit-learn.
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
        Initialize the SKLearn model wrapper.

        Args:
            name (str): The model name.
            version (str): The model version.
            model_type (SupportedModelTypes): The type of the model (regression or classification).
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
            framework=SupportedMLFrameworks.SKL,
        )
        self.model = None

    def save_model(self, path: os.PathLike = None, *args, **kwargs) -> None:
        """
        Save the scikit-learn model to the specified file path using cloudpickle.

        Args:
            path (os.PathLike): The file path where the model should be saved.
            *args: Additional positional arguments for cloudpickle.dump.
            **kwargs: Additional keyword arguments for cloudpickle.dump.

        Raises:
            ValueError: If there is no model instance available to save.
            Exception: Propagates any exception raised during file writing.
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
            with local_path.open("wb") as f:
                cloudpickle.dump(self.model, f, *args, **kwargs)
            logger.info("SKLearn model saved successfully to %s", local_path)
        except Exception:
            logger.exception("Failed to save SKLearn model to %s", local_path)
            raise

    def load_model(self, path: os.PathLike = None, *args, **kwargs) -> "BaseModel":
        """
        Load a scikit-learn model from the specified file path using cloudpickle.

        If the provided path is a directory, the default model file path is used.

        Args:
            path (os.PathLike): The file path (or directory) from which to load the model.
            *args: Additional positional arguments for cloudpickle.load.
            **kwargs: Additional keyword arguments for cloudpickle.load.

        Returns:
            BaseModel: Self, to allow method chaining.

        Raises:
            Exception: Propagates any exception raised during file reading.
        """
        local_path = self.default_model_filepath() if path is None else Path(path)
        try:
            with local_path.open("rb") as f:
                self.model = cloudpickle.load(f, *args, **kwargs)
            logger.info("SKLearn model loaded successfully from %s", local_path)
            return self
        except Exception:
            logger.exception("Failed to load SKLearn model from %s", local_path)
            raise

    def predict_score(self, data: npt.NDArray) -> npt.NDArray:
        """
        Compute prediction scores for the given input data.

        For regression models, returns the raw predictions.
        For classification models, returns the probability of the positive class.

        Args:
            data (npt.NDArray): Input data array (1D or 2D).

        Returns:
            npt.NDArray: Array of prediction scores.
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
            logger.exception("Error during predict_score.")
            raise

    def predict(self, data: npt.NDArray) -> ModelOutputs:
        """
        Generate predictions for the given input data.

        For regression, only predictions are produced.
        For classification, predictions and probabilities are produced.

        Args:
            data (npt.NDArray): Input data array (1D or 2D).

        Returns:
            ModelOutputs: An object containing predictions and probabilities (if applicable).
        """
        one_entry = data.ndim == 1
        data = data.reshape(1, -1) if one_entry else data
        pred_outs = ModelOutputs()
        try:
            if self.is_regression:
                pred_outs.predictions = self.model.predict(data)
            else:
                pred_outs.predictions = self.model.predict(data)
                pred_outs.probabilities = self.model.predict_proba(data)[:, 1]
            return pred_outs
        except Exception:
            logger.exception("Error during predict.")
            raise
