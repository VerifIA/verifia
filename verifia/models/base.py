import os
import logging
from typing import Iterable, Optional, Union, List, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy.typing as npt
from sklearn.metrics import root_mean_squared_error, f1_score

from ..context.data import Dataset
from ..mlops import get_configured_mlops_platform
from ..mlops import comet_ml, mlflow, wandb
from ..utils.enums.models import (
    SupportedMLFrameworks,
    SupportedModelTypes,
    ModelMetricNames,
)
from ..utils.enums.mlops import SupportedMLOpsPlatform
from ..utils.helpers.io import (
    mk_tmpdir,
    rm_tmpdir,
    first_file_dot_ext_in_folder,
    save_yaml,
)
from ..utils import DEFAULT_MODEL_EXTS

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class ModelCard:
    """
    Data class representing a model card that stores metadata about the model.

    Attributes:
        name (str): Model name.
        version (str): Model version.
        framework (str): Framework used to build the model.
        description (Optional[str]): Model description.
        model_type (SupportedModelTypes): Type of the model (e.g., classification, regression).
        feature_names (List[str]): List of feature names used by the model.
        cat_feature_names (Optional[List[str]]): Optional list of categorical feature names.
        target_name (str): Name of the target variable.
        classification_threshold (float): Threshold used for classification.
    """

    name: str
    version: str
    framework: SupportedMLFrameworks
    description: Optional[str]
    model_type: SupportedModelTypes
    feature_names: List[str]
    cat_feature_names: Optional[List[str]]
    target_name: str
    classification_threshold: float
    local_dirpath: str

    @property
    def full_name(self) -> str:
        """
        Return a full name combining model name and version.
        """
        return f"{self.name}-{self.version}"

    def to_dict(self) -> dict:
        default_dict = self.__dict__.copy()

        default_dict["model_type"] = self.model_type.value
        default_dict["framework"] = self.framework.value
        return default_dict


@dataclass
class ModelOutputs:
    """
    Container for model prediction outputs.

    Attributes:
        predictions (Optional[npt.NDArray]): Array of predictions.
        probabilities (Optional[npt.NDArray]): Array of probabilities (if available).
    """

    predictions: Optional[npt.NDArray] = None
    probabilities: Optional[npt.NDArray] = None


class BaseModel(ABC):
    """
    Abstract base class for wrapping ML models.

    Provides common functionality such as model card handling,
    saving/loading from a registry, and performance evaluation.
    """

    def __init__(
        self,
        name: str,
        version: str,
        framework: Union[str, SupportedMLFrameworks],
        model_type: Union[str, SupportedModelTypes],
        feature_names: Iterable,
        target_name: str,
        local_dirpath: str,
        cat_feature_names: Optional[Iterable] = None,
        classification_threshold: Optional[float] = 0.5,
        description: Optional[str] = None,
    ) -> None:
        """
        Initialize the BaseModel with the required metadata and validations.

        Args:
            name (str): Model name.
            version (str): Model version.
            framework (Union[str, SupportedMLFrameworks]): Model framework.
            model_type (Union[str, SupportedModelTypes]): Model type.
            feature_names (Iterable): Iterable of feature names.
            target_name (str): Target variable name.
            cat_feature_names (Optional[Iterable]): Optional iterable of categorical feature names.
            classification_threshold (Optional[float]): Threshold for classification (default 0.5).
            description (Optional[str]): Model description.

        Raises:
            ValueError: If the provided model_type or framework values are invalid,
                        or if any categorical feature is not in the feature names.
        """
        # Validate and convert model_type to enum if provided as string.
        if isinstance(model_type, str):
            try:
                model_type = SupportedModelTypes(model_type)
            except ValueError as e:
                available_values = {i.value for i in SupportedModelTypes}
                raise ValueError(
                    f'Invalid model type value "{model_type}". Available values are: {available_values}'
                ) from e

        # Validate and convert framework to enum if provided as string.
        if isinstance(framework, str):
            try:
                framework = SupportedMLFrameworks(framework)
            except ValueError as e:
                available_values = {i.value for i in SupportedMLFrameworks}
                raise ValueError(
                    f'Invalid framework value "{framework}". Available values are: {available_values}'
                ) from e

        # Validate that each categorical feature is included in the feature names.
        if cat_feature_names is not None:
            cat_feature_names = list(cat_feature_names)
            for cat_feature_name in cat_feature_names:
                if cat_feature_name not in feature_names:
                    raise ValueError(
                        f"Categorical feature '{cat_feature_name}' is not included in feature_names: {list(feature_names)}"
                    )

        self.model_card = ModelCard(
            name=name,
            version=version,
            framework=framework,
            description=description if description is not None else "No description",
            model_type=model_type,
            feature_names=list(feature_names),
            cat_feature_names=(
                list(cat_feature_names) if cat_feature_names is not None else None
            ),
            target_name=target_name,
            local_dirpath=local_dirpath,
            classification_threshold=classification_threshold,
        )

    def set_name_n_version(self, name: str, version: str) -> None:
        """
        Update the model card's name and version.

        Args:
            name (str): New model name.
            version (str): New model version.
        """
        self.model_card.name = name
        self.model_card.version = version

    @property
    def name(self) -> str:
        """
        Get the model's name.
        """
        return self.model_card.name

    @property
    def version(self) -> str:
        """
        Get the model's version.
        """
        return self.model_card.version

    @property
    def framework(self) -> SupportedMLFrameworks:
        """
        Get the model's framework.
        """
        return self.model_card.framework

    @property
    def description(self) -> str:
        """
        Get the model's description.
        """
        return self.model_card.description

    @property
    def dirpath(self) -> str:
        return self.model_card.local_dirpath

    @property
    def model_type(self) -> SupportedModelTypes:
        """
        Get the model's type.
        """
        return self.model_card.model_type

    @property
    def feature_names(self) -> List[str]:
        """
        Get the list of feature names.
        """
        return self.model_card.feature_names

    @property
    def cat_feature_names(self) -> Optional[List[str]]:
        """
        Get the list of categorical feature names.
        """
        return self.model_card.cat_feature_names

    @property
    def target_name(self) -> str:
        """
        Get the target variable name.
        """
        return self.model_card.target_name

    @property
    def classification_threshold(self) -> float:
        """
        Get the classification threshold.
        """
        return self.model_card.classification_threshold

    @property
    def is_classification(self) -> bool:
        """
        Check if the model is for classification.
        """
        return self.model_type == SupportedModelTypes.CLASSIFICATION

    @property
    def is_regression(self) -> bool:
        """
        Check if the model is for regression.
        """
        return self.model_type == SupportedModelTypes.REGRESSION

    @property
    def has_model(self) -> bool:
        """
        Check if there is a wrapped model.
        """
        return self.model is not None

    def save_model_card(self, local_path: os.PathLike) -> None:
        """
        Save the model card as a YAML file to the specified path.

        Args:
            local_path (os.PathLike): File path where the model card will be saved.
        """
        save_yaml(local_path, self.model_card.to_dict())
        logger.info("Model card saved to %s", local_path)

    @abstractmethod
    def save_model(self, path: os.PathLike = None) -> None:
        """
        Save the wrapped model object to the specified path.
        Must be implemented by subclasses.

        Args:
            path (os.PathLike): Destination file path.
        """
        raise NotImplementedError("Subclasses should implement save_model().")

    def wrap_model(self, model: Any) -> "BaseModel":
        """
        Wrap the provided model.

        This method sets the given model as the instance's model and returns the instance,
        allowing for method chaining.

        Args:
            model (Any): The model to be wrapped.

        Returns:
            BaseModel: The instance with the model attribute set.
        """
        self.model = model
        return self

    @abstractmethod
    def load_model(self, path: Union[str, Path] = None, *args, **kwargs) -> "BaseModel":
        """
        Load the wrapped model object from the specified path.
        Must be implemented by subclasses.

        Args:
            path (Union[str, Path]): File path from where to load the model.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The loaded model object.
        """
        raise NotImplementedError("Subclasses should implement load_model().")

    def load_model_from_registry(
        self, model_ext: Optional[str] = None, *args, **kwargs
    ) -> None:
        """
        Load the model from the MLOps registry based on the configured platform.

        The method uses the current model's name, version, and framework to load the model.
        Depending on the configured MLOps platform, the model is either loaded directly via MLflow
        or downloaded temporarily via Comet ML or WandB.

        Args:
            model_ext (Optional[str]): Optional file extension override. If not provided, the default
                                       extension for the model's framework is used.

        Raises:
            ValueError: If the model name or version is not set or if the MLOps platform is unsupported.
        """
        if self.name is None or self.version is None:
            raise ValueError(
                "Both 'name' and 'version' attributes must be set to load the model from registry."
            )

        mlops_platform = get_configured_mlops_platform()
        if model_ext is None:
            model_ext = DEFAULT_MODEL_EXTS[self.framework]

        logger.info(
            "Loading model '%s' version '%s' using platform '%s'",
            self.name,
            self.version,
            mlops_platform.value,
        )
        if mlops_platform == SupportedMLOpsPlatform.MLFLOW:
            # Directly load the model via MLflow.
            self.model = mlflow.load_model(self.name, self.version, self.framework)
        elif mlops_platform == SupportedMLOpsPlatform.COMET_ML:
            tmp_dirpath = mk_tmpdir()
            comet_ml.download_model(self.name, self.version, tmp_dirpath)
            local_path = first_file_dot_ext_in_folder(tmp_dirpath, model_ext)
            self.load_model(local_path, *args, **kwargs)
            rm_tmpdir(tmp_dirpath)
        elif mlops_platform == SupportedMLOpsPlatform.WANDB:
            tmp_dirpath = mk_tmpdir()
            wandb.download_model(self.name, self.version, tmp_dirpath)
            local_path = first_file_dot_ext_in_folder(tmp_dirpath, model_ext)
            self.load_model(local_path, *args, **kwargs)
            rm_tmpdir(tmp_dirpath)
        else:
            raise ValueError(f"Unsupported MLOps platform: {mlops_platform}")

    def calculate_predictive_performance(self, dataset: Dataset) -> Tuple[str, float]:
        """
        Calculate the predictive performance of the model on the provided dataset.

        For classification, the F1 score is computed.
        For regression, the Root Mean Squared Error (RMSE) is computed.

        Args:
            dataset (Dataset): An object with attributes 'X' (features) and 'y' (targets).

        Returns:
            Tuple[str, float]: A tuple containing the metric name and the computed score.

        Raises:
            ValueError: If the model type is unknown.
        """
        outs = self.predict(dataset.X)
        if self.is_classification:
            score = f1_score(dataset.y, outs.predictions)
            return ModelMetricNames.F1_SCORE.value, score
        elif self.is_regression:
            rmse = root_mean_squared_error(dataset.y, outs.predictions)
            return ModelMetricNames.RMSE.value, rmse
        else:
            raise ValueError("Unknown model type for performance evaluation.")

    @abstractmethod
    def predict_score(self, data: Any) -> npt.NDArray:
        """
        Compute and return prediction scores for the provided data.
        Must be implemented by subclasses.

        Args:
            data (Any): Input data for which to compute prediction scores.

        Returns:
            npt.NDArray: An array of prediction scores.
        """
        raise NotImplementedError("Subclasses should implement predict_score().")

    @abstractmethod
    def predict(self, data: Any) -> ModelOutputs:
        """
        Perform predictions on the provided data.
        Must be implemented by subclasses.

        Args:
            data (Any): Input data for prediction.

        Returns:
            ModelOutputs: An object containing predictions and, if applicable, probabilities.
        """
        raise NotImplementedError("Subclasses should implement predict().")

    def default_model_filepath(self, other_local_dirpath: str = None) -> Path:
        """
        Construct a default file path for saving the model, based on the model's name and framework.

        Returns:
            str: The full file path including the default file extension.
        """
        default_model_fext = DEFAULT_MODEL_EXTS[self.framework]
        model_filename = f"{self.name}-{self.version}.{default_model_fext}"
        if other_local_dirpath is None:
            model_filepath = os.path.join(self.dirpath, model_filename)
        else:
            model_filepath = os.path.join(other_local_dirpath, model_filename)
        logger.info("Default model file path constructed: %s", model_filepath)
        return Path(model_filepath)
