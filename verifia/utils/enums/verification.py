from enum import Enum


class SearchAlgorithm(str, Enum):
    RS = "RS"
    FFA = "FFA"
    MFO = "MFO"
    GWO = "GWO"
    MVO = "MVO"
    PSO = "PSO"
    WOA = "WOA"
    GA = "GA"
    SSA = "SSA"
