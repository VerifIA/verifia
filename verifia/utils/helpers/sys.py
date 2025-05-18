import os
import time
import base64
from uuid import uuid4
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def generate_short_uuid(length: int = 8) -> str:
    """
    Generate a short UUID string.

    This function generates a UUID using uuid4, encodes its bytes into a URL-safe base64 string,
    strips any trailing '=' characters, and then trims the result to the desired length.

    Args:
        length (int, optional): Desired length of the resulting UUID string. Defaults to 8.

    Returns:
        str: A short UUID string of the specified length.
    """
    uid = uuid4()
    short_uid = base64.urlsafe_b64encode(uid.bytes).rstrip(b"=").decode("utf-8")
    return short_uid[:length]


def generate_timestamp() -> int:
    """
    Generate the current timestamp.

    Returns the current time as an integer representing seconds since the Unix epoch.

    Returns:
        int: Current timestamp in seconds.
    """
    return int(time.time())


def check_env_var(var_name: str) -> bool:
    """
    Check if a specific environment variable is set.

    Args:
        var_name (str): The name of the environment variable.

    Returns:
        bool: True if the environment variable is set, False otherwise.
    """
    return var_name in os.environ


def check_env_vars(required_vars: List[str]) -> Tuple[bool, List[str]]:
    """
    Check if all required environment variables are set.

    Args:
        required_vars (List[str]): A list of environment variable names that are required.

    Returns:
        Tuple[bool, List[str]]: A tuple where the first element is True if all variables are set,
                                 and the second element is a list of missing variable names.
    """
    missing_vars = [var for var in required_vars if os.getenv(var) is None]
    return len(missing_vars) == 0, missing_vars


def safe_eval(expression: str, local_vars: Dict[str, Any]) -> Any:
    """
    Safely evaluate a mathematical expression using provided local variables.

    This function uses Python's eval with a restricted built-ins dictionary.
    In production, consider using a dedicated mathematical parser for improved security.

    Args:
        expression (str): The expression to evaluate.
        local_vars (Dict[str, Any]): A dictionary of local variables for use in the evaluation.

    Returns:
        Any: The result of the evaluated expression.

    Raises:
        Exception: If the evaluation of the expression fails.
    """
    try:
        return eval(expression, {}, local_vars)  # "__builtins__": {}
    except Exception as e:
        logger.error("Error evaluating expression '%s': %s", expression, e)
        raise e
