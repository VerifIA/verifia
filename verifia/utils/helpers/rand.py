from typing import List, Union
import numpy as np


def uniform_excluding(
    a: float, b: float, exclude: Union[float, List[float]], tol: float = 1e-6
) -> float:
    """
    Generate a random float from a uniform distribution within [a, b], excluding specified values.

    The function generates a random float in [a, b] (with b made inclusive using np.finfo(float).eps)
    and excludes values that, when rounded to a precision determined by `tol`, match any value in
    `exclude`.

    Args:
        a (float): Lower bound of the uniform distribution (inclusive).
        b (float): Upper bound of the uniform distribution (inclusive, adjusted with epsilon).
        exclude (Union[float, List[float]]): Value(s) to exclude.
        tol (float, optional): Tolerance for floating point comparisons. Default is 1e-6.

    Returns:
        float: A random float within [a, b] that does not equal any excluded value (within tolerance).
    """
    decimals = int(-np.log10(tol))

    if isinstance(exclude, list):
        norm_exclude = {
            round(val, decimals) if isinstance(val, float) else val for val in exclude
        }
    else:
        norm_exclude = (
            round(exclude, decimals) if isinstance(exclude, float) else exclude
        )

    while True:
        value = np.random.uniform(a, b + np.finfo(float).eps)
        norm_value = round(value, decimals) if isinstance(value, float) else value
        if isinstance(exclude, list):
            if norm_value not in norm_exclude:
                return value
        else:
            if norm_value != norm_exclude:
                return value


def randint_excluding(a: int, b: int, exclude: Union[int, List[int]]) -> int:
    """
    Generate a random integer from a uniform distribution within [a, b], excluding specified values.

    The function generates a random integer in [a, b] (inclusive) and excludes any integer
    present in the `exclude` parameter.

    Args:
        a (int): Lower bound of the integer range (inclusive).
        b (int): Upper bound of the integer range (inclusive).
        exclude (Union[int, List[int]]): Integer(s) to exclude.

    Returns:
        int: A random integer in [a, b] that is not equal to any excluded value.
    """
    if isinstance(exclude, list):
        norm_exclude = set(exclude)
    else:
        norm_exclude = exclude

    while True:
        value = np.random.randint(a, b + 1)
        if isinstance(exclude, list):
            if value not in norm_exclude:
                return value
        else:
            if value != norm_exclude:
                return value


def uniform_including(
    a: float, b: float, include: Union[float, List[float]], tol: float = 1e-6
) -> float:
    """
    Generate a random float from a uniform distribution within [a, b] that equals one of the specified values.

    This function repeatedly generates a random float until it (when rounded according to `tol`)
    exactly matches one of the values in `include`. Use with caution as this may loop indefinitely
    if the probability is extremely low.

    Args:
        a (float): Lower bound of the uniform distribution (inclusive).
        b (float): Upper bound of the uniform distribution (inclusive, adjusted with epsilon).
        include (Union[float, List[float]]): The value(s) that must be generated.
        tol (float, optional): Tolerance for floating point comparisons. Default is 1e-6.

    Returns:
        float: A random float within [a, b] that equals one of the included values (within tolerance).
    """
    decimals = int(-np.log10(tol))

    if isinstance(include, list):
        norm_include = {
            round(val, decimals) if isinstance(val, float) else val for val in include
        }
    else:
        norm_include = (
            round(include, decimals) if isinstance(include, float) else include
        )

    while True:
        value = np.random.uniform(a, b + np.finfo(float).eps)
        norm_value = round(value, decimals) if isinstance(value, float) else value
        if isinstance(include, list):
            if norm_value in norm_include:
                return value
        else:
            if norm_value == norm_include:
                return value


def randint_including(a: int, b: int, include: Union[int, List[int]]) -> int:
    """
    Generate a random integer from a uniform distribution within [a, b] that equals one of the specified values.

    The function repeatedly generates a random integer until it matches one of the integers
    provided in `include`.

    Args:
        a (int): Lower bound of the integer range (inclusive).
        b (int): Upper bound of the integer range (inclusive).
        include (Union[int, List[int]]): The integer(s) that must be generated.

    Returns:
        int: A random integer in [a, b] that equals one of the included values.
    """
    if isinstance(include, list):
        norm_include = set(include)
    else:
        norm_include = include

    while True:
        value = np.random.randint(a, b + 1)
        if isinstance(include, list):
            if value in norm_include:
                return value
        else:
            if value == norm_include:
                return value
