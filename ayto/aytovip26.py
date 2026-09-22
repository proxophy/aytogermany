from .ayto import Solver, get_candidates
from .models import *

import itertools


class VIP2026Solver(Solver):
    sm: str = "Laurenz"
    double_match_for_right: bool = True

    def check_multiple_match_logic(self, sol: Solution, end: int) -> dict:

        if not super().check_multiple_match_logic(sol, end)["res"]:
            return super().check_multiple_match_logic(sol, end)

        pdict = {r: [] for r in self.season.rights}
        for l, r in sol:
            pdict[r].append(l)

        mutiplers = [
            r for r in self.season.rights if len(pdict[r]) > 1
        ]  # rights with multiple matches
        if len(mutiplers) > 1:
            return {"res": False, "reason": "more_than_one_double_right_match"}
        elif len(mutiplers) == 1:
            multir = mutiplers[0]
            multils = pdict[multir]
            if len(multils) == 2 and self.sm not in multils:
                return {"res": False, "reason": "Laurenz_not_in_double_right_match"}

        return {"res": True, "reason": ""}

    def generate_complete_solutions(
        self, sol: Solution, end: int, include_night: bool
    ) -> list[Solution]:
        if not self.solution_possible(sol, end, include_night)["res"]:
            return []

        if (
            len(sol) == self.season.num_matches
            and self.solution_possible(sol, end, include_night)["res"]
        ):
            return [sol]

        def zip_product(clefts, ordering) -> Solution:
            return Solution(p for p in zip(clefts, ordering))

        glefts, _, mmnum = get_candidates(sol)

        pos_matches = self.possible_matches_for_solution(sol, end, include_night)
        pos_matches.pop(self.sm, None)  # type: ignore
        if self.sm in glefts:
            # remove pair with self.sm if necessary
            smmatches = [r for l, r in sol if l == self.sm]
            # print(smmatches)
            # isol = set(sol)
            sol = sol.difference({(self.sm, smmatches[0])})
        else:
            smmatches = [
                r
                for r in self.season.rights
                if not self.no_match(self.sm, r, end)  # type: ignore
            ]

        products = [
            list(ps)
            for ps in itertools.product(*pos_matches.values())
            if len(set(ps)) == len(ps)
        ]
        partial_solutions = list(
            map(lambda p: zip_product(pos_matches.keys(), p), products)
        )

        if mmnum > 0:
            isols = self.merge_mm_in_solution(sol, partial_solutions)
        else:
            isols = self.merge_mm_not_in_solution(
                sol, partial_solutions, end, include_night
            )

        # add self.sm as double_match
        sitting_no_matches = self.season.get_sitting_no_matches(sol, end, include_night)
        solutions = [
            s | {(self.sm, r)}
            for r in smmatches
            for s in isols
            if (self.sm, r) not in sitting_no_matches
        ] 

        return solutions
