import os
from pathlib import Path
from typing import Optional, Iterable
import numpy as np
import numpy.typing as npt
from .base import BaseModel, ModelOutputs
from ..utils.enums.models import SupportedModelTypes, SupportedMLFrameworks
from ..utils.helpers.optional import require
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MissingBuildFloatingTensorsError(AttributeError):
    """
    Raised when the 'build_floating_tensors' function is missing in torch.nn.Module.

    This function is responsible for converting input data into floating-point tensors,
    ensuring categorical values are properly mapped to numerical representations before processing.
    """

    def __init__(self):
        message = (
            "Your torch.nn.Module must implement the 'build_floating_tensors' function.\n"
            "This function is responsible for converting input data into floating-point tensors, "
            "ensuring categorical values are properly mapped to numerical representations before processing."
        )
        super().__init__(message)


class PytorchModel(BaseModel):
    """
    A PyTorch model wrapper that implements the BaseModel interface.

    Provides methods to save, load, and perform inference with PyTorch models.
    Expects the underlying model to implement a 'build_floating_tensors' method for preprocessing.
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
        Initialize the PyTorch model wrapper.

        Args:
            name (str): The model name.
            version (str): The model version.
            model_type (SupportedModelTypes): Model type (classification or regression).
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
            framework=SupportedMLFrameworks.PTH,
        )
        self.model = None
        self._torch = require("torch", "torch")

    def save_model(self, path: os.PathLike = None) -> None:
        
        """
        Save the PyTorch model to the specified file path.

        Args:
            path (os.PathLike): The path (or directory) to save the model.
        """
        if path is None:
            local_path = self.default_model_filepath()
        else:
            local_path = Path(path)

        local_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise ValueError("No model to save. Load or train a model first.")
        try:
            self._torch.save(self.model, local_path)
            logger.info("PyTorch model saved successfully to %s", local_path)
        except Exception:
            logger.exception("Failed to save PyTorch model to %s", local_path)
            raise

    def load_model(self, path: os.PathLike = None) -> "BaseModel":
        """
        Load the PyTorch model from the specified file path.

        If the provided local_path is a directory, the default model filepath is used.

        Args:
            path (os.PathLike): The path (or directory) from which to load the model.

        Returns:
            BaseModel: Self, to allow method chaining.

        Raises:
            MissingBuildFloatingTensorsError: If the loaded model does not implement 'build_floating_tensors'.
            Exception: Propagates any exception raised during model loading.
        """
        local_path = self.default_model_filepath() if path is None else Path(path)
        try:
            self.model = self._torch.load(local_path, weights_only=False)
            if not hasattr(self.model, "build_floating_tensors"):
                raise MissingBuildFloatingTensorsError()
            logger.info("PyTorch model loaded successfully from %s", local_path)
            return self
        except Exception:
            logger.exception("Failed to load PyTorch model from %s", local_path)
            raise

    def predict_score(self, data: npt.NDArray) -> npt.NDArray:
        """
        Compute prediction scores for the given input data.

        For regression, returns raw predictions.
        For classification, returns probability scores or class scores as appropriate.

        Args:
            data (npt.NDArray): Input data array (1D or 2D).

        Returns:
            npt.NDArray: Array of prediction scores.

        Raises:
            Exception: Propagates any exception raised during prediction.
        """
        self.model.eval()
        one_entry = data.ndim == 1
        data = data.reshape(1, -1) if one_entry else data

        try:
            # Preprocess input data using the model's helper method.
            data_tensor = self.model.build_floating_tensors(data)
        except Exception:
            logger.exception("Error in building floating tensors from data.")
            raise

        with self._torch.no_grad():
            scores = self.model(data_tensor)

        if self.is_regression:
            result = scores.cpu().detach().numpy().ravel()
        else:
            if scores.dim() > 1 and scores.size(-1) > 1:
                result = self._torch.softmax(scores, dim=-1).cpu().detach().numpy()[:, 1]
            else:
                result = self._torch.sigmoid(scores).cpu().detach().numpy().ravel()

        return result[0] if one_entry and result.ndim == 1 else result

    def predict(self, data: npt.NDArray) -> ModelOutputs:
        """
        Generate predictions for the input data.

        For regression, only predictions are generated.
        For classification, generates predictions, probabilities, and maps predictions to labels.

        Args:
            data (npt.NDArray): Input data array (1D or 2D).

        Returns:
            ModelOutputs: Object containing predictions and probabilities (if applicable).

        Raises:
            Exception: Propagates any exception raised during prediction.
        """
        self.model.eval()
        one_entry = data.ndim == 1
        data = data.reshape(1, -1) if one_entry else data

        try:
            data_tensor = self.model.build_floating_tensors(data)
        except Exception:
            logger.exception("Error in building floating tensors from data.")
            raise

        pred_outs = ModelOutputs()
        with self._torch.no_grad():
            scores = self.model(data_tensor)

        if self.is_regression:
            pred_outs.predictions = scores.cpu().detach().numpy().ravel()
        else:
            if scores.dim() > 1 and scores.size(-1) > 1:
                probabilities = self._torch.softmax(scores, dim=1).cpu().detach().numpy()
                pred_outs.probabilities = probabilities[:, 1]
                pred_outs.predictions = (
                    self._torch.argmax(probabilities, dim=1).cpu().detach().numpy()
                )
            else:
                probabilities = self._torch.sigmoid(scores).cpu().detach().numpy().ravel()
                pred_outs.probabilities = probabilities
                pred_outs.predictions = np.int32(
                    probabilities > self.classification_threshold
                )
        return pred_outs
