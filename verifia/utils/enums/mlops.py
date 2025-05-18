from enum import Enum


class SupportedMLOpsPlatform(str, Enum):
    """
    Enumeration of supported MLOps platforms.
    """

    COMET_ML = "comet_ml"
    MLFLOW = "mlflow"
    WANDB = "wandb"
