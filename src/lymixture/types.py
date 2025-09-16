"""Type definitions for the lymixture package."""

from typing import Literal

ICDCode = Literal[
    "C00",  # Lip
    "C01",  # Base of tongue
    "C02",  # Other and unspecified parts of tongue
    "C03",  # Gum
    "C04",  # Floor of mouth
    "C05",  # Palate
    "C06",  # Other and unspecified parts of mouth
    "C07",  # Parotid gland
    "C08",  # Other and unspecified major salivary gland
    "C09",  # Tonsil
    "C10",  # Oropharynx
    "C11",  # Nasopharynx
    "C12",  # Pyriform sinus
    "C13",  # Hypopharynx
    "C14",  # Other and ill-defined sites in lip, oral cavity and pharynx
    "C30",  # Nasal cavity and middle ear
    "C31",  # Accessory sinuses
    "C32",  # Larynx
    "C73",  # Thyroid gland
]
