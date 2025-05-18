from enum import Enum


class ChangeType(Enum):
    """Enumeration of allowed change types for variables."""

    INC = "inc"
    DEC = "dec"
    NOINC = "noinc"
    NODEC = "nodec"
    CST = "cst"
    VAR = "var"
    EQ = "eq"
    NOEQ = "noeq"
    IN = "in"
    NOIN = "noin"


class VarType(Enum):
    """Enumeration of variable types."""

    INT = "INT"
    FLOAT = "FLOAT"
    CAT = "CAT"
