"""Module with utilities for the mixture model package."""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.special import factorial, softmax

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)

RESP_COLS = ("_mixture", "responsibility")

def binom_pmf(k: np.ndarray, n: int, p: float) -> np.ndarray:
    """Compute binomial PMF fast."""
    if p > 1.0 or p < 0.0:
        msg = "Binomial prob must be btw. 0 and 1"
        raise ValueError(msg)
    q = 1.0 - p
    binom_coeff = factorial(n) / (factorial(k) * factorial(n - k))
    return binom_coeff * p**k * q ** (n - k)


def late_binomial(support: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Parametrized binomial distribution."""
    return binom_pmf(k=support, n=support[-1], p=p)


def map_to_simplex(from_real: np.ndarray | list[float]) -> np.ndarray:
    """Map from real numbers to simplex.

    The result has one entry more than ``values``.
    The method creates a simplex by adding a dimension which is fixed to zero
    Then the values are run through a softmax function to normalize them.

    >>> real = [4, 3.5]
    >>> map_to_simplex(real)
    array([0.01127223, 0.61544283, 0.37328494])
    """
    non_normalized = np.array([0.0, *from_real])
    return softmax(non_normalized, axis=0)


def map_to_real(from_simplex: np.ndarray | list[float]) -> np.ndarray:
    """Map from simplex to real numbers.

    >>> simplex = [0.01127223, 0.61544283, 0.37328494]
    >>> np.allclose(map_to_real(simplex), [4, 3.5])
    True
    """
    from_simplex = np.array(from_simplex)
    normalizer = 1 / from_simplex[0]
    return np.log(from_simplex[1:] * normalizer)


def normalize(
    values: np.ndarray,
    axis: int,
    **isclose_kwargs: float | bool,
) -> np.ndarray:
    """Normalize ``values`` to sum to 1 along ``axis``.

    Beyond normalizing, this function also sets values that are close to zero to the
    exact value of zero. For this, it passes all extra keyword arguments to numpy's
    ``isclose`` function.

    >>> normalize(np.array([0.1, 0.2, 0.7]), axis=0)    # doctest: +NORMALIZE_WHITESPACE
    array([0.1, 0.2, 0.7])
    >>> normalize(np.array([1e-20, 0.3, 0.7]), axis=0)  # doctest: +NORMALIZE_WHITESPACE
    array([0. , 0.3, 0.7])
    >>> normalize(np.array([1e-20, 0.3, 0.3]), axis=0)  # doctest: +NORMALIZE_WHITESPACE
    array([0. , 0.5, 0.5])
    """
    normalized = values / np.sum(values, axis=axis)
    small_idx = np.isclose(normalized, 0.0, **isclose_kwargs)
    normalized[small_idx] = 0.0
    return normalized / np.sum(normalized, axis=axis)


def log_normalize(
    log_values: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Log-normalize ``log_values`` to sum to 1 along ``axis``.

    >>> log_norm = log_normalize(np.log([0.1, 0.2, 0.7]), axis=0)
    >>> np.exp(np.logaddexp.reduce(log_norm, axis=0))   # doctest: +NORMALIZE_WHITESPACE
    np.float64(1.0)
    """
    return log_values - np.logaddexp.reduce(log_values, axis=axis)


def harden(values: np.ndarray, axis: int) -> np.ndarray:
    """Harden ``values`` to become a one-hot-encoding along the given ``axis``.

    >>> values = np.array(
    ...     [[0.1, 0.2, 0.7],
    ...      [0.3, 0.4, 0.3]]
    ... )
    >>> harden(values, axis=1)   # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0, 1],
           [0, 1, 0]])
    >>> arr = np.array([[[0.84, 0.64, 0.3 , 0.23],
    ...                  [0.18, 0.31, 0.23, 0.54],
    ...                  [0.08, 0.05, 0.72, 0.09]],
    ...                 [[0.33, 0.43, 0.28, 0.54],
    ...                  [0.26, 0.48, 0.8 , 0.01],
    ...                  [0.45, 0.09, 0.64, 0.11]]])
    >>> harden(arr, axis=2)      # doctest: +NORMALIZE_WHITESPACE
    array([[[1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]],
           [[0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 0]]])
    >>> harden(np.array([0.1, 0.2, 0.3, 0.1]), axis=0)
    array([0, 0, 1, 0])
    """
    maxdim = len(values.shape) - 1
    idx = np.argmax(values, axis=axis)  # one dim less than `values`
    one_hot = np.eye(values.shape[axis], dtype=int)[idx]  # right dim, but wrong order
    dim_sort = (*range(axis), maxdim, *range(axis, maxdim))
    return one_hot.transpose(*dim_sort)  # right order


def join_with_resps(
    patient_data: pd.DataFrame,
    num_components: int,
    resps: np.ndarray | None = None,
) -> pd.DataFrame:
    """Join patient data with empty responsibilities (and reset index)."""
    mixture_columns = pd.MultiIndex.from_tuples(
        [(*RESP_COLS, i) for i in range(num_components)],
    )

    if resps is None:
        resps = np.empty(shape=(len(patient_data), num_components))
        resps.fill(np.nan)
        resps = pd.DataFrame(resps, columns=mixture_columns)

    if RESP_COLS in patient_data:
        patient_data = patient_data.drop(columns=RESP_COLS)

    return patient_data.join(resps).reset_index(drop=True)


def one_slice(idx: int) -> slice:
    """Return a slice that selects only one element at the given index.

    Helpful if one wants to select one element, but return it as a list.

    >>> l = [1, 2, 3, 4, 5]
    >>> l[one_slice(2)]
    [3]
    """
    return slice(idx, idx + 1)