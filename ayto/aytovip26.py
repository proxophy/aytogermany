from .ayto import AYTO
from .models import *

import itertools


class AYTOVIP2026(AYTO):
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
        self.sm = "Laurenz"
        self.nummatches = 12

    def no_match(self, p: Pair, options: dict[str, bool]) -> bool:
        return super().no_match(p, options)

    def solution_correct_format(self, sol: Solution) -> bool:
        if len(sol) > self.nummatches:
            print("len(solution) > self.nummatches")
            return False

        ls, rs = zip(*sol)
        if any(l not in self.lefts for l in ls):
            print(f"Wrongly written names: {[c for c in ls if c not in self.lefts]}")
            return False
        if any(r not in self.rights for r in rs):
            print(f"Wrongly written names: {[c for c in rs if c not in self.rights]}")
            return False
       
        return True

    def solution_possible(self, sol: Solution, options: dict) -> bool:

        if not super().solution_possible(sol, options):
            return False

        pdict = {r: [] for r in self.rights}
        for l, r in sol:
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


    def possible_matches_for_solution(
        self, sol: Solution, options: dict
    ) -> dict[str, list[str]]:
        return super().possible_matches_for_solution(sol, options)

    def merge_mm_not_in_solution(
        self, sol: Solution, other_matches_list: list[Solution], options: dict
    ):
        return super().merge_mm_not_in_solution(sol, other_matches_list, options)

    def generate_complete_solutions(
        self, sol: Solution, options: dict
    ) -> list[Solution]:
        pass

        if len(sol) == self.nummatches and self.solution_possible(sol, options):
            return [sol]
        # print("in aytovip26")

        def zip_product(clefts, ordering):
            return set(Pair(*p) for p in zip(clefts, ordering))

        glefts, _, dm_in_psol = self.get_solution_leftrights(sol)
        pos_matches = self.possible_matches_for_solution(sol, options)
        pos_matches.pop(self.sm)
        smr = None
        if self.sm in glefts:
            # remove pair with self.sm if necessary
            smmatches = [r for l, r in sol if l == self.sm][0]
            sol.remove(Pair(self.sm, smmatches[0]))
        else:
            smmatches = [
                r for r in self.rights if not self.no_match(Pair(self.sm, r), options)
            ]

        # print(smmatches)

        products = [
            list(ps)
            for ps in itertools.product(*pos_matches.values())
            if len(set(ps)) == len(ps)
        ]
        other_matches_list = list(
            map(lambda p: Solution(zip_product(pos_matches.keys(), p)), products)
        )



        if dm_in_psol > 0:
            isols = self.merge_mm_in_solution(sol, other_matches_list, options)
        else:
            isols = self.merge_mm_not_in_solution(sol, other_matches_list, options)
            # add self.sm as double_match
        solutions = [s.addpair(Pair(self.sm, r)) for r in smmatches for s in isols]
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
    # season: AYTO = AYTOVIP2026(*utils.read_data_from_excel(sn))

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

    # sols = season.generate_complete_solutions(psol, options)
    # print("tsol in sols", tsol in sols)
    # print("subset", psol.issubset(tsol))
    # print(season.partialsol_correct_format(tsol))

    # csols = season.generate_complete_solutions(psol, options)
    # print("len(csols)", len(csols))
    # df = utils.sols_as_df(csols)
    # df = df.sort_values(by=season.lefts[:-1])[season.lefts]
    # print(df)
