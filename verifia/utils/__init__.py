import os
import shutil
from .enums.models import SupportedMLFrameworks
from .enums.models import SupportedModelTypes

ROUNDING_PRECISION = os.getenv("VERIFIA_ROUNDING_PRECISION", 2)
CHECKPOINTS_DIRPATH = os.getenv("VERIFIA_CHECKPOINTS_DIRPATH", ".verifia")

EPSILON = 1e-12
VERIFIA_REPORTS = "verifia-reports"
DEFAULT_MODEL_EXTS = {
    SupportedMLFrameworks.SKL: "pkl",
    SupportedMLFrameworks.LGBM: "txt",
    SupportedMLFrameworks.CB: "cb",
    SupportedMLFrameworks.XGB: "json",
    SupportedMLFrameworks.PTH: "pth",
    SupportedMLFrameworks.TF: "keras",
}
ML_FMKS_NAME_VARIATIONS = {
    SupportedMLFrameworks.SKL: ["sklearn", "scikit-learn", "skl"],
    SupportedMLFrameworks.LGBM: ["lgb", "lgbm", "lightgbm"],
    SupportedMLFrameworks.CB: ["cb", "catboost"],
    SupportedMLFrameworks.XGB: ["xgb", "xgboost"],
    SupportedMLFrameworks.PTH: ["pth", "pytorch", "torch"],
    SupportedMLFrameworks.TF: ["keras", "tf", "tensorflow"],
}
ML_MODEL_TYPE_VARIATIONS = {
    SupportedModelTypes.REGRESSION: ["regressor", "regression", "reg"],
    SupportedModelTypes.CLASSIFICATION: ["classifier", "classification", "cls"],
}


def clean_cache():
    shutil.rmtree(str(CHECKPOINTS_DIRPATH))
