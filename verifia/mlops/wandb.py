import os
import logging
from ..utils.helpers.sys import check_env_var
from ..utils.helpers.io import validate_dirpath, validate_filepath
from ..utils import VERIFIA_REPORTS
from ..utils.helpers.optional import require

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def build_registered_model_uri(name: str, version: str) -> str:
    """
    Build the WandB artifact URI for a registered model.

    Args:
        name (str): The name of the model.
        version (str): The version of the model.

    Returns:
        str: A string representing the WandB artifact URI.
    """
    model_uri = f"{name}:v{version}"
    logger.debug("Built model URI: %s", model_uri)
    return model_uri


def download_model(
    model_name: str, model_version: str, model_dirpath: os.PathLike
) -> None:
    """
    Download a model artifact from Weights & Biases and save it locally.

    This function requires the following environment variables to be defined:
        - WANDB_API_KEY
        - WANDB_PROJECT
        - WANDB_ENTITY

    Args:
        model_name (str): The name of the registered model.
        model_version (str): The version of the model.
        model_dirpath (os.PathLike): Local directory path where the model should be downloaded.

    Raises:
        EnvironmentError: If any required environment variable is not defined.
        Exception: If there is an error during model download.
    """
    wandb = require("wandb", "wandb")
    required_vars = ["WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY"]
    for var in required_vars:
        if not check_env_var(var):
            raise EnvironmentError(f"Environment variable '{var}' is not defined.")

    validate_dirpath(model_dirpath)

    try:
        wandb.init()  # Ensure wandb is initialized
        model_uri = build_registered_model_uri(model_name, model_version)
        logger.info("Downloading model artifact with URI: %s", model_uri)
        model_artifact = wandb.use_artifact(model_uri)
        model_artifact.download(model_dirpath)
        logger.info(
            "Model '%s' version '%s' downloaded to '%s'.",
            model_name,
            model_version,
            model_dirpath,
        )
    except Exception:
        logger.exception(
            "Failed to download model '%s' version '%s'.", model_name, model_version
        )
        raise


def get_model_artifact(model_name: str, model_version: str):
    """
    Retrieve a model artifact from Weights & Biases.

    This function requires the following environment variables to be defined:
        - WANDB_API_KEY
        - WANDB_PROJECT
        - WANDB_ENTITY

    Args:
        model_name (str): The name of the registered model.
        model_version (str): The version of the model.

    Returns:
        wandb.Artifact: The WandB Artifact object representing the model.

    Raises:
        EnvironmentError: If any required environment variable is not defined.
        Exception: If there is an error retrieving the model artifact.
    """
    wandb = require("wandb", "wandb")
    required_vars = ["WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY"]
    for var in required_vars:
        if not check_env_var(var):
            raise EnvironmentError(f"Environment variable '{var}' is not defined.")

    try:
        wandb.init()  # Ensure wandb is initialized
        api = wandb.Api()
        model_uri = build_registered_model_uri(model_name, model_version)
        logger.info("Retrieving model artifact with URI: %s", model_uri)
        model_artifact = api.artifact(model_uri)
        return model_artifact
    except Exception:
        logger.exception(
            "Failed to retrieve model artifact for '%s' version '%s'.",
            model_name,
            model_version,
        )
        raise


def log_report(model_name: str, model_version: str, local_fpath: os.PathLike) -> None:
    """
    Log a report file as an artifact to Weights & Biases.

    The function creates a report artifact with metadata referencing the model artifact.

    Args:
        model_name (str): The name of the model.
        model_version (str): The version of the model.
        local_fpath (os.PathLike): The local file path of the report to log.

    Raises:
        Exception: If an error occurs while logging the report artifact.
    """
    wandb = require("wandb", "wandb")
    validate_filepath(local_fpath)
    try:
        model_artifact = get_model_artifact(model_name, model_version)
        report_artifact = wandb.Artifact(name=VERIFIA_REPORTS, type="report")
        report_artifact.add_file(local_fpath)
        report_artifact.metadata = {
            "model-fullname": f"{model_artifact.entity}/{model_artifact.project}/{model_artifact.name}",
            "model-artifact-id": model_artifact.id,
        }
        wandb.log_artifact(report_artifact)
        logger.info(
            "Logged report artifact for model '%s' version '%s'.",
            model_name,
            model_version,
        )
    except Exception:
        logger.exception(
            "Failed to log report for model '%s' version '%s'.",
            model_name,
            model_version,
        )
        raise
