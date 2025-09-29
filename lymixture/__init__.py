"""A mixture model for lymphatic metastasis in head and neck cancer.

This package defines classes and functions to model lymphatic involvement as a
mixture of hidden Markov models as they are implemented in the :py:mod:`lymph` package.
"""

import logging

from lymixture._version import version
from lymixture.em import expectation, maximization
from lymixture.models import LymphMixture

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["LymphMixture", "expectation", "maximization"]
__version__ = version
__author__ = "Roman Ludwig"
__email__ = "gygqdstu3@mozmail.com"
__uri__ = "https://github.com/lycosystem/lymixture"
