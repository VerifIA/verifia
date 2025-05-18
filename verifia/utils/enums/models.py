from enum import Enum


class SupportedMLFrameworks(str, Enum):
    """
    Supported machine learning frameworks for loading models via MLflow.
    """

    SKL = "sklearn"
    LGBM = "lightgbm"
    CB = "catboost"
    XGB = "xgboost"
    PTH = "pytorch"
    TF = "tensorflow"


class SupportedModelTypes(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ModelMetricNames(str, Enum):
    F1_SCORE = "F1-Score"
    RMSE = "RMSE"
