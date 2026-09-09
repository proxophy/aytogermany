from .ayto import AYTO
from .models import *

class AYTOVIP2023(AYTO):
    def __init__(
        self,
        lefts: list[str],
        rights: list[str],
        nights: list[Night],
        matchboxes: Matchboxes = Matchboxes(),
        dm: str | None = None,
        solution: Solution | None = None,
    ) -> None:
        super().__init__(lefts, rights, nights, matchboxes, dm, solution)
        self.dmtuple = ("Peter", "Max")
