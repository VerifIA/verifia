import os
import logging
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import numpy.typing as npt
from .base import BaseModel, ModelOutputs
from ..utils.enums.models import SupportedModelTypes, SupportedMLFrameworks
from ..utils.helpers.optional import require

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TFModel(BaseModel):
    """
    A TensorFlow/Keras model wrapper that implements the BaseModel interface.

    This class provides methods to save, load, and perform predictions with Keras models.
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
        Initialize the TFModel.

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
            framework=SupportedMLFrameworks.TF,
        )
        self._tf = require("tensorflow", "tensorflow")
        self.model = None

    def save_model(self, path: os.PathLike = None, *args, **kwargs) -> None:
        """
        Save the Keras model to the specified path.

        Args:
            path (os.PathLike): The file path where the model should be saved.
            *args: Additional positional arguments for the Keras save method.
            **kwargs: Additional keyword arguments for the Keras save method.

        Raises:
            Exception: Propagates exceptions raised during saving.
        """
        if path is None:
            local_path = self.default_model_filepath()
        else:
            local_path = Path(path)

        local_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise ValueError("No model to save. Load or train a model first.")
        try:
            self.model.save(str(local_path), *args, **kwargs)
            logger.info("Keras model saved successfully to %s", local_path)
        except Exception:
            logger.exception("Failed to save Keras model to %s", local_path)
            raise

    def load_model(self, path: os.PathLike = None, *args, **kwargs) -> "BaseModel":
        """
        Load a Keras model from the specified path.

        If the provided local_path is a directory, the default model file path is used.

        Args:
            path (os.PathLike): The file path (or directory) from which to load the model.
            *args: Additional positional arguments for load_model.
            **kwargs: Additional keyword arguments for load_model.

        Returns:
            BaseModel: Self, to allow method chaining.

        Raises:
            Exception: Propagates exceptions raised during loading.
        """
        local_path = self.default_model_filepath() if path is None else Path(path)
        try:
            self.model = self._tf.keras.models.load_model(str(local_path), *args, **kwargs)
            logger.info("Keras model loaded successfully from %s", local_path)
            return self
        except Exception:
            logger.exception("Failed to load Keras model from %s", local_path)
            raise

    def predict_score(self, data: npt.NDArray) -> npt.NDArray:
        """
        Predict probabilities or regression scores for the given input data.

        Assumes a single output for the model.

        Args:
            data (npt.NDArray): Input data array (1D or 2D).

        Returns:
            npt.NDArray: Array of prediction scores.

        Raises:
            Exception: Propagates exceptions raised during data conversion or prediction.
        """
        one_entry = data.ndim == 1
        data = data.reshape(1, -1) if one_entry else data

        try:
            # Prepare inputs as a dictionary of tensors, one per feature.
            inputs_dict = {
                name: self._tf.convert_to_tensor([row[idx] for row in data])
                for idx, name in enumerate(self.feature_names)
            }
        except Exception:
            logger.exception("Error while converting input data to tensors.")
            raise

        try:
            scores = self.model.predict(inputs_dict, verbose=0)
        except Exception:
            logger.exception("Error during model prediction.")
            raise

        if self.is_regression:
            return scores.ravel()
        else:
            if scores.ndim > 1 and scores.shape[-1] > 1:
                probabilities = self._tf.nn.softmax(scores, axis=1).numpy()
                return probabilities[:, 1]
            else:
                return self._tf.math.sigmoid(scores).numpy().ravel()

    def predict(self, data: npt.NDArray) -> ModelOutputs:
        """
        Generate predictions for the given data.

        For classification, includes probabilities and mapped labels.
        For regression, includes only predictions.

        Args:
            data (npt.NDArray): Input data array (1D or 2D).

        Returns:
            ModelOutputs: An object containing predictions, probabilities, and labels (if applicable).

        Raises:
            Exception: Propagates exceptions raised during data conversion or prediction.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        try:
            inputs_dict = {
                name: self._tf.convert_to_tensor([row[idx] for row in data])
                for idx, name in enumerate(self.feature_names)
            }
        except Exception:
            logger.exception("Error while converting input data to tensors.")
            raise

        pred_outs = ModelOutputs()
        try:
            scores = self.model.predict(inputs_dict, verbose=0)
        except Exception:
            logger.exception("Error during model prediction.")
            raise

        if self.is_regression:
            pred_outs.predictions = scores.ravel()
        else:
            if scores.ndim > 1 and scores.shape[-1] > 1:
                probabilities = self._tf.nn.softmax(scores, axis=1).numpy()
                pred_outs.probabilities = probabilities[:, 1]
                pred_outs.predictions = self._tf.argmax(scores, axis=1).numpy().ravel()
            else:
                probabilities = self._tf.math.sigmoid(scores).numpy().ravel()
                pred_outs.probabilities = probabilities
                pred_outs.predictions = np.int32(
                    probabilities >= self.classification_threshold
                )
        return pred_outs
