from .ayto import AYTO

from typing import Optional, Union
from collections import Counter
import pandas as pd
import ayto.utils as utils
import itertools

PartialSol = set[tuple[str, str]]
CompleteSol = set[tuple[str, str]]
Night = tuple[list[tuple[str, str]], int]
Matchboxes = dict[tuple[str, str], bool]


class AYTOVIP2026(AYTO):
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
        self.sm = "Laurenz"
        self.nummatches = 12

    def no_match(self, l: str, r: str, options: dict[str, bool]) -> bool:
        return super().no_match(l, r, options)

    def partialsol_correct_format(self, psol: PartialSol) -> bool:
        if len(psol) > self.nummatches:
            print("len(solution) > self.nummatches")
            return False

        ls, rs = zip(*psol)
        if any(l not in self.lefts for l in ls):
            print(f"Wrongly written names: {[c for c in ls if c not in self.lefts]}")
            return False
        if any(r not in self.rights for r in rs):
            print(f"Wrongly written names: {[c for c in rs if c not in self.rights]}")
            return False
       
        return True

    def partialsol_possible(self, psol: PartialSol, options: dict) -> bool:

        if not super().partialsol_possible(psol, options):
            return False

        pdict = {r: [] for r in self.rights}
        for l, r in psol:
            pdict[r].append(l)

        mutiplers = [
            r for r in self.rights if len(pdict[r]) > 1
        ]  # rights with multiple matches
        if len(mutiplers) > 1:
            # print(f"more than one right has multiple matches: {mutiplers}")
            return False
        elif len(mutiplers) == 1:
            multir = mutiplers[0]
            multils = pdict[multir]
            if len(multils) == 2 and self.sm not in multils:
                # print(f"{self.sm} must be one of the double matches for right {multir} {multils}")
                return False

        return True


    def possible_matches_for_partialsol(
        self, psol: PartialSol, options: dict
    ) -> dict[str, list[str]]:
        return super().possible_matches_for_partialsol(psol, options)

    def merge_mm_not_in_partialsol(
        self, psol: PartialSol, other_matches_list: list[PartialSol], options: dict
    ):
        return super().merge_mm_not_in_partialsol(psol, other_matches_list, options)

    def generate_complete_solutions(
        self, psol: PartialSol, options: dict
    ) -> list[CompleteSol]:
        pass

        if len(psol) == self.nummatches and self.partialsol_possible(psol, options):
            return [psol]
        # print("in aytovip26")

        def zip_product(clefts, ordering):
            return set(zip(clefts, ordering))

        glefts, _, dm_in_psol = self.get_partialsol_leftrights(psol)
        pos_matches = self.possible_matches_for_partialsol(psol, options)
        pos_matches.pop(self.sm)
        smr = None
        if self.sm in glefts:
            # remove pair with self.sm if necessary
            smmatches = [r for l, r in psol if l == self.sm][0]
            psol.remove((self.sm, smmatches[0]))
        else:
            smmatches = [
                r for r in self.rights if not self.no_match(self.sm, r, options)
            ]

        # print(smmatches)

        products = [
            list(ps)
            for ps in itertools.product(*pos_matches.values())
            if len(set(ps)) == len(ps)
        ]
        other_matches_list = list(
            map(lambda p: zip_product(pos_matches.keys(), p), products)
        )

        # print("dm_in_psol", dm_in_psol)
        # print("len(products)", len(products))
        # print("len(other_matches_list)", len(other_matches_list))

        if dm_in_psol > 0:
            isols = self.merge_mm_in_partialsol(psol, other_matches_list, options)
        else:
            isols = self.merge_mm_not_in_partialsol(psol, other_matches_list, options)
            # add self.sm as double_match
        solutions = [s.union({(self.sm, r)}) for r in smmatches for s in isols]
        unique_sols = []

        for sol in solutions:
            assert (
                len(sol) == self.nummatches
            ), f"Complete solutions with {self.nummatches} pairs, not {len(sol)} pairs "

            if sol not in unique_sols:
                unique_sols.append(sol)

        return unique_sols


if __name__ == "__main__":
    sn = "vip2026"
    options = {"end": 2, "includenight": True, "verbose": False}
    season: AYTO = AYTOVIP2026(*utils.read_data_from_excel(sn))

    psol = {
        ("Robin", "Christin"),
        ("Raúl", "Alexandra"),
        ("Marwin", "Joena"),
        ("Fabian", "Julia"),
        ("Bennet", "Francesca"),
        ("Johannes", "Marta"),
        ("Johannes", "Janice"),
    }

    tsol = {
        ("Johannes", "Janice"),
        ("Marwin", "Joena"),
        ("Raúl", "Alexandra"),
        ("Bennet", "Francesca"),
        ("Cansin", "Michelle"),
        ("Robin", "Christin"),
        ("Germain", "Jenny"),
        ("Fabian", "Julia"),
        ("Johannes", "Marta"),
        ("Daymian", "Emma"),
        ("Brian", "Zoe"),
        ("Laurenz", "Joena"),
    }

    sols = season.generate_complete_solutions(psol, options)
    print("tsol in sols", tsol in sols)
    print("subset", psol.issubset(tsol))
    print(season.partialsol_correct_format(tsol))

    # csols = season.generate_complete_solutions(psol, options)
    # print("len(csols)", len(csols))
    # df = utils.sols_as_df(csols)
    # df = df.sort_values(by=season.lefts[:-1])[season.lefts]
    # print(df)
