from typing import Type, Any, Dict, Union
from .base import BaseModel
from .base import ModelCard
from ..utils.enums.models import SupportedMLFrameworks, SupportedModelTypes
from .sklearn import SKLearnModel
from .tensorflow import TFModel
from .pytorch import PytorchModel
from .xgb import XGBModel
from .lgb import LGBModel
from .cb import CBModel
from ..utils.helpers.io import read_yaml
from ..utils import ML_FMKS_NAME_VARIATIONS
from ..utils import ML_MODEL_TYPE_VARIATIONS


def get_selected_model_type(model_type_str: str) -> SupportedModelTypes:
    """
    Normalize and match the input model type string to a d value.

    Args:
        model_type_str (str): A string representing the ML Model Type.

    Returns:
        SupportedModelTypes: The matched Model Type enum.

    Raises:
        ValueError: If the model type is unsupported.
    """
    normalized_value = model_type_str.strip().lower()
    for model_type, variations in ML_MODEL_TYPE_VARIATIONS.items():
        if normalized_value in variations:
            return model_type
    raise ValueError(f"Unsupported ML Model Type: {model_type_str}")


def get_selected_ml_framework(model_fmk_str: str) -> SupportedMLFrameworks:
    """
    Normalize and match the input framework string to a SupportedMLFrameworks value.

    Args:
        model_fmk_str (str): A string representing the ML framework.

    Returns:
        SupportedMLFrameworks: The matched ML framework enum.

    Raises:
        ValueError: If the framework is unsupported.
    """
    normalized_value = model_fmk_str.strip().lower()
    for framework, variations in ML_FMKS_NAME_VARIATIONS.items():
        if normalized_value in variations:
            return framework
    raise ValueError(f"Unsupported ML framework: {model_fmk_str}")


def get_class_name(model_fmk: SupportedMLFrameworks) -> Type[BaseModel]:
    """
    Get the model class corresponding to the provided ML framework.

    Args:
        model_fmk (SupportedMLFrameworks): The ML framework.

    Returns:
        Type[BaseModel]: A class object that is a subclass of BaseModel.

    Raises:
        ValueError: If the framework is unsupported.
    """
    fmk_class_mappings = {
        SupportedMLFrameworks.SKL: SKLearnModel,
        SupportedMLFrameworks.LGBM: LGBModel,
        SupportedMLFrameworks.CB: CBModel,
        SupportedMLFrameworks.XGB: XGBModel,
        SupportedMLFrameworks.PTH: PytorchModel,
        SupportedMLFrameworks.TF: TFModel,
    }
    if model_fmk not in fmk_class_mappings:
        raise ValueError(f"Unsupported ML framework: {model_fmk}")
    return fmk_class_mappings[model_fmk]


def build_from_model_card(model_card_input: Union[str, Dict[str, Any]]) -> BaseModel:
    """
    Build a model instance from a model card.

    This function accepts either a file path (str) to a YAML model card or a model card
    directly provided as a dictionary. The model card must contain at least the following keys:

        - name: The model name.
        - version: The model version.
        - framework: The machine learning framework used.
        - type: The model type.
        - feature_names: A list of feature names.
        - target_name: The target variable name.
    
    Optionally, the model card may contain:

        - cat_feature_names: A list of categorical feature names.
        - classification_threshold: A threshold for classification tasks.
        - description: A description of the model.
        - local_dirpath: The local directory path for the model.

    Args:
        model_card_input (Union[str, Dict[str, Any]]): Either the file path to the model card YAML file
            or the model card dictionary itself.

    Returns:
        BaseModel: An instance of a model (a subclass of BaseModel) configured as per the model card.

    Raises:
        FileNotFoundError: If the specified model card file does not exist (when given a file path).
        KeyError: If any required key is missing from the model card.
        yaml.YAMLError: If the YAML file cannot be parsed.
        ValueError: If the framework specified in the model card is not supported.
    """
    if isinstance(model_card_input, str):
        model_card = read_yaml(model_card_input)
    else:
        model_card = model_card_input

    try:
        name = model_card["name"]
        version = model_card["version"]
        framework = get_selected_ml_framework(model_card["framework"])
        model_type = get_selected_model_type(model_card["type"])
        feature_names = model_card["feature_names"]
        target_name = model_card["target_name"]
        local_dirpath = model_card["local_dirpath"]
        cat_feature_names = model_card.get("cat_feature_names")
        classification_threshold = model_card.get("classification_threshold", 0.5)
        description = model_card.get("description")
    except KeyError as e:
        raise KeyError(f"Missing required key in model card: {e}")

    model_cls = get_class_name(framework)
    return model_cls(
        name=name,
        version=version,
        model_type=model_type,
        feature_names=feature_names,
        target_name=target_name,
        local_dirpath=local_dirpath,
        cat_feature_names=cat_feature_names,
        classification_threshold=classification_threshold,
        description=description,
    )
