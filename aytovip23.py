from ayto import AYTO, time_it

from typing import Optional, Union
from collections import Counter
import pandas as pd
import utils
import itertools

PartialSol = set[tuple[str, str]]
CompleteSol = set[tuple[str, str]]
Night = tuple[list[tuple[str, str]], int]
Matchboxes = dict[tuple[str, str], bool]


class AYTOVIP2023(AYTO):
    def __init__(
        self,
        lefts: list[str],
        rights: list[str],
        nights: list[tuple[list[tuple[str, str]], int]],
        enmatchboxes: dict[tuple[int, str, str], bool] = {},
        dm: str | None = None,
        solution: Optional[set[tuple[str, str]]] = None,
    ) -> None:
        super().__init__(lefts, rights, nights, enmatchboxes, dm, solution)
        self.dmtuple = ("Peter", "Max")
