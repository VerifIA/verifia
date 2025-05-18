import logging
import random as rand
import numpy as np
import numpy.typing as npt
from .. import ROUNDING_PRECISION

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def compute_percentage(divide: float, whole: float) -> float:
    """
    Compute the percentage of 'divide' with respect to 'whole', rounded to a defined precision.

    Args:
        divide (float): The numerator value.
        whole (float): The denominator value.

    Returns:
        float: The percentage value (0 if whole is zero).
    """
    if whole == 0:
        return 0.0
    return round(100 * divide / whole, ROUNDING_PRECISION)


def softmax(P: npt.NDArray, axis: int = -1) -> npt.NDArray:
    """
    Compute the softmax function in a numerically stable way.

    Args:
        P (npt.NDArray): Input array (1D, 2D, or higher).
        axis (int): Axis along which to apply softmax. Default is the last axis.

    Returns:
        npt.NDArray: Softmax-normalized probabilities.
    """
    P = np.asarray(P, dtype=np.float64)
    P_max = np.max(P, axis=axis, keepdims=True)  # Prevent overflow
    e_P = np.exp(P - P_max)
    return e_P / np.sum(e_P, axis=axis, keepdims=True)


def normalization(M: npt.NDArray) -> npt.NDArray:
    """
    Normalize a vector (1D) or each row of a 2D matrix to unit L2 norm.

    For a 1D array, returns a normalized vector with the same shape.
    For a 2D array, each row is normalized independently and the output has the same shape.

    Args:
        M (npt.NDArray): Input array (1D or 2D) to normalize.

    Returns:
        npt.NDArray: Normalized array with unit L2 norm for each vector.

    Raises:
        ValueError: If the input array is not 1D or 2D.
    """
    if M.ndim == 1:
        norm = np.linalg.norm(M)
        if norm == 0:
            norm = 1
        return M / norm
    elif M.ndim == 2:
        norm = np.linalg.norm(M, axis=1, keepdims=True)
        norm[norm == 0] = 1  # Prevent division by zero for any zero-vector row
        return M / norm
    else:
        raise ValueError("Normalization supports only 1D or 2D arrays.")


def roulette_wheel_selection(weights: npt.NDArray) -> int:
    """
    Select an index based on a weighted probability distribution using the roulette wheel method.

    Args:
        weights (npt.NDArray): Array of weights representing the probabilities of selection.

    Returns:
        int: The index of the selected element.
    """
    accumulation = np.cumsum(weights)
    selection_prob = rand.random() * accumulation[-1]
    for idx, val in enumerate(accumulation):
        if val > selection_prob:
            return idx
    return -1
