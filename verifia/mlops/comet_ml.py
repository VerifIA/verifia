import os
import logging
from ..utils.helpers.sys import check_env_var
from ..utils.helpers.io import validate_dirpath, validate_filepath
from ..utils import VERIFIA_REPORTS
from ..utils.helpers.optional import require

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def download_model(
    model_name: str, model_version: str, model_dirpath: os.PathLike
) -> None:
    """
    Download a model from Comet.ml and save it to a local directory.

    This function requires the following environment variables to be defined:
      - COMET_API_KEY
      - COMET_WORKSPACE
      - COMET_PROJECT_NAME

    Args:
        model_name (str): Name of the model to download.
        model_version (str): Version of the model to download.
        model_dirpath (os.PathLike): Local directory path where the model will be saved.

    Raises:
        EnvironmentError: If any required environment variable is not defined.
        Exception: If an error occurs during model download.
    """
    comet_ml = require("comet_ml", "comet_ml")
    required_env_vars = ["COMET_API_KEY", "COMET_WORKSPACE", "COMET_PROJECT_NAME"]
    for var in required_env_vars:
        if not check_env_var(var):
            raise EnvironmentError(f"Environment variable '{var}' is not defined.")

    validate_dirpath(model_dirpath)

    try:
        api = comet_ml.API()
        workspace = os.getenv("COMET_WORKSPACE")
        model = api.get_model(workspace=workspace, model_name=model_name)
        model.download(model_version, model_dirpath)
        logger.info(
            f"Downloaded model '{model_name}' version '{model_version}' to '{model_dirpath}'."
        )
    except Exception:
        logger.exception("Failed to download model.")
        raise


def get_model_experiment_id(model_name: str, model_version: str) -> str:
    """
    Retrieve the experiment ID associated with a given model version from Comet.ml.

    This function requires the following environment variables to be defined:
      - COMET_API_KEY
      - COMET_WORKSPACE
      - COMET_PROJECT_NAME

    Args:
        model_name (str): Name of the model.
        model_version (str): Version of the model.

    Returns:
        str: The experiment ID.

    Raises:
        EnvironmentError: If any required environment variable is not defined.
        ValueError: If no assets are found or the experiment ID is missing.
        Exception: If an error occurs while retrieving the experiment ID.
    """
    comet_ml = require("comet_ml", "comet_ml")
    required_env_vars = ["COMET_API_KEY", "COMET_WORKSPACE", "COMET_PROJECT_NAME"]
    for var in required_env_vars:
        if not check_env_var(var):
            raise EnvironmentError(f"Environment variable '{var}' is not defined.")

    try:
        api = comet_ml.API()
        workspace = os.getenv("COMET_WORKSPACE")
        model = api.get_model(workspace=workspace, model_name=model_name)
        assets = model.get_assets(version=model_version)
        if not assets:
            raise ValueError(
                f"No assets found for model '{model_name}' version '{model_version}'."
            )
        experiment_id = assets[0].get("experimentKey")
        if not experiment_id:
            raise ValueError("Experiment key not found in model assets.")
        logger.info(
            f"Retrieved experiment ID '{experiment_id}' for model '{model_name}' version '{model_version}'."
        )
        return experiment_id
    except Exception:
        logger.exception("Failed to retrieve model experiment ID.")
        raise


def log_report(model_name: str, model_version: str, local_fpath: os.PathLike) -> None:
    """
    Log a report file to an existing Comet.ml experiment.

    This function retrieves the experiment ID for the specified model and version,
    then logs the file as an asset to that experiment with associated metadata.

    Args:
        model_name (str): Name of the model.
        model_version (str): Version of the model.
        local_fpath (os.PathLike): Local file path of the report to be logged.

    Raises:
        Exception: If an error occurs while logging the report.
    """
    comet_ml = require("comet_ml", "comet_ml")
    validate_filepath(local_fpath)

    try:
        experiment_id = get_model_experiment_id(model_name, model_version)
        existing_experiment = comet_ml.ExistingExperiment(
            previous_experiment=experiment_id
        )
        existing_experiment.log_asset(
            local_fpath,
            step=0,
            log_file_name=False,
            metadata={
                "report": VERIFIA_REPORTS,
                "model": f"{model_name}-v{model_version}",
            },
        )
        logger.info(
            f"Logged report '{local_fpath}' for model '{model_name}' version '{model_version}'."
        )
    except Exception:
        logger.exception("Failed to log report.")
        raise
