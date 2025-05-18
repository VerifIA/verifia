import os
import logging
from typing import Any
from ..utils.helpers.sys import check_env_var
from ..utils.helpers.io import validate_filepath
from ..utils.enums.models import SupportedMLFrameworks
from ..utils import VERIFIA_REPORTS
from ..utils.helpers.optional import require

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def build_registered_model_uri(name: str, version: str) -> str:
    """
    Build the MLflow registered model URI using the model name and version.

    Args:
        name (str): The registered model name.
        version (str): The registered model version.

    Returns:
        str: The constructed model URI.
    """
    model_uri = f"models:/{name}/{version}"
    logger.debug("Built model URI: %s", model_uri)
    return model_uri


def load_model(
    model_name: str, model_version: str, model_framework: SupportedMLFrameworks
) -> Any:
    """
    Load a model from the MLflow Model Registry based on its framework.

    Requires the environment variable 'MLFLOW_TRACKING_URI' to be set.

    Args:
        model_name (str): The registered model name.
        model_version (str): The registered model version.
        model_framework (SupportedMLFrameworks): The ML framework used to train the model.

    Returns:
        Any: The loaded model object.

    Raises:
        EnvironmentError: If 'MLFLOW_TRACKING_URI' is not defined.
        ValueError: If an unsupported ML framework is specified.
        Exception: If an error occurs during model loading.
    """
    mlflow = require("mlfow", "mlflow")

    if not check_env_var("MLFLOW_TRACKING_URI"):
        raise EnvironmentError(
            "Environment variable 'MLFLOW_TRACKING_URI' is not defined."
        )

    model_uri = build_registered_model_uri(model_name, model_version)
    logger.info(
        "Loading model '%s' version '%s' using framework '%s'",
        model_name,
        model_version,
        model_framework,
    )

    try:
        if model_framework == SupportedMLFrameworks.SKL:
            loaded_model = mlflow.sklearn.load_model(model_uri)
        elif model_framework == SupportedMLFrameworks.LGBM:
            loaded_model = mlflow.lightgbm.load_model(model_uri)
        elif model_framework == SupportedMLFrameworks.CB:
            loaded_model = mlflow.catboost.load_model(model_uri)
        elif model_framework == SupportedMLFrameworks.XGB:
            loaded_model = mlflow.xgboost.load_model(model_uri)
        elif model_framework == SupportedMLFrameworks.PTH:
            loaded_model = mlflow.pytorch.load_model(model_uri)
        elif model_framework == SupportedMLFrameworks.TF:
            loaded_model = mlflow.tensorflow.load_model(model_uri)
        else:
            raise ValueError(f"Unsupported model framework: {model_framework}")
    except Exception:
        logger.exception("Failed to load model from URI: %s", model_uri)
        raise

    logger.info("Model loaded successfully from URI: %s", model_uri)
    return loaded_model


def get_registered_model_run_id(model_name: str, model_version: str) -> str:
    """
    Retrieve the MLflow run ID for a registered model.

    Requires the environment variable 'MLFLOW_TRACKING_URI' to be set.

    Args:
        model_name (str): The registered model name.
        model_version (str): The registered model version.

    Returns:
        str: The run ID associated with the registered model version.

    Raises:
        EnvironmentError: If 'MLFLOW_TRACKING_URI' is not defined.
        Exception: If an error occurs while retrieving the run ID.
    """
    mlflow = require("mlfow", "mlflow")

    if not check_env_var("MLFLOW_TRACKING_URI"):
        raise EnvironmentError(
            "Environment variable 'MLFLOW_TRACKING_URI' is not defined."
        )

    try:
        client = mlflow.tracking.MlflowClient()
        model_metadata = client.get_model_version(model_name, model_version)
        run_id = model_metadata.run_id
        logger.info(
            "Retrieved run ID '%s' for model '%s' version '%s'",
            run_id,
            model_name,
            model_version,
        )
        return run_id
    except Exception:
        logger.exception(
            "Failed to retrieve model run ID for model '%s' version '%s'",
            model_name,
            model_version,
        )
        raise


def log_report(model_name: str, model_version: str, local_fpath: os.PathLike) -> None:
    """
    Log a report artifact to the MLflow run corresponding to a registered model.

    The report is logged as an artifact under the path specified by VERIFIA_REPORTS.

    Args:
        model_name (str): The registered model name.
        model_version (str): The registered model version.
        local_fpath (os.PathLike): The local file path of the report to log.

    Raises:
        Exception: If an error occurs while logging the artifact.
    """
    mlflow = require("mlfow", "mlflow")
    
    validate_filepath(local_fpath)

    try:
        model_run_id = get_registered_model_run_id(model_name, model_version)
        logger.info("Logging report '%s' to run ID '%s'", local_fpath, model_run_id)
        with mlflow.start_run(run_id=model_run_id):
            mlflow.log_artifact(local_fpath, artifact_path=VERIFIA_REPORTS)
        logger.info(
            "Report logged successfully for model '%s' version '%s'",
            model_name,
            model_version,
        )
    except Exception:
        logger.exception(
            "Failed to log report for model '%s' version '%s'",
            model_name,
            model_version,
        )
        raise
