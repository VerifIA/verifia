import logging
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from a .env file, if available.
load_dotenv()

from ..utils.helpers.sys import check_env_vars
from ..utils.enums.mlops import SupportedMLOpsPlatform

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_configured_mlops_platform() -> SupportedMLOpsPlatform:
    """
    Determine which MLOps platform is fully configured based on required environment variables.

    The function checks the following platforms and their corresponding environment variables:
      - MLflow: requires 'MLFLOW_TRACKING_URI'
      - Comet ML: requires 'COMET_API_KEY', 'COMET_WORKSPACE', 'COMET_PROJECT_NAME'
      - WandB: requires 'WANDB_API_KEY', 'WANDB_PROJECT', 'WANDB_ENTITY'

    If multiple platforms are fully configured, a warning is logged and the first platform is returned.
    If no platform is fully configured, errors are logged for each partially configured platform and
    a ValueError is raised.

    Returns:
        SupportedMLOpsPlatform: The fully configured MLOps platform.

    Raises:
        ValueError: If no fully configured MLOps platform is found.
    """
    platform_requirements: Dict[SupportedMLOpsPlatform, List[str]] = {
        SupportedMLOpsPlatform.MLFLOW: ["MLFLOW_TRACKING_URI"],
        SupportedMLOpsPlatform.COMET_ML: [
            "COMET_API_KEY",
            "COMET_WORKSPACE",
            "COMET_PROJECT_NAME",
        ],
        SupportedMLOpsPlatform.WANDB: [
            "WANDB_API_KEY",
            "WANDB_PROJECT",
            "WANDB_ENTITY",
        ],
    }

    configured_platforms: List[SupportedMLOpsPlatform] = []
    missing_info: Dict[SupportedMLOpsPlatform, List[str]] = {}

    for platform, env_vars in platform_requirements.items():
        all_set, missing_vars = check_env_vars(env_vars)
        if all_set:
            configured_platforms.append(platform)
        else:
            missing_info[platform] = missing_vars

    if len(configured_platforms) > 1:
        logger.warning(
            "Multiple MLOps platforms are configured: %s. Please configure only one platform to avoid conflicts.",
            ", ".join(platform.value for platform in configured_platforms),
        )
    elif not configured_platforms:
        for platform, missing_vars in missing_info.items():
            logger.error(
                "Platform '%s' is partially configured. Missing environment variables: %s.",
                platform.value,
                ", ".join(missing_vars),
            )
        raise ValueError("No fully configured MLOps platform found.")

    selected_platform = configured_platforms[0]
    logger.info("Configured MLOps platform to be used: %s", selected_platform.value)
    return selected_platform
