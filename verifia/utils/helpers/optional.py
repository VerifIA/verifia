import importlib

def require(module: str, extra: str):
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"Install the `{extra}` extra: `pip install verifia[{extra}]`"
        )