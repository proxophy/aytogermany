from .ayto import AYTO, Pair, Night, Matchboxes, Solution
from .aytonormalo24 import AYTONormalo2024
from .aytovip25 import AYTOVIP2025
from .aytovip23 import AYTOVIP2023
from .aytovip26 import AYTOVIP2026
from .solver import find_solutions
from .analysis import sol_probs

import functools
import time

__all__ = [
    "AYTO",
    "AYTONormalo2024",
    "AYTOVIP2025",
    "AYTOVIP2023",
    "AYTOVIP2026",
    "find_solutions",
    "Pair",
    "Solution",
    "Night",
    "Matchboxes",
]



