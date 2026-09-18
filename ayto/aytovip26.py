from .ayto import Solver, get_candidates
from .models import *

import itertools


class VIP2026Solver(Solver):
    sm: str = "Laurenz"

    def solution_correct_format(self, sol: Solution) -> dict:
        if len(sol) > self.season.num_matches:
            return {"res": False, "reason": "too_many_matches"}

        ls, rs = zip(*sol)
        if any(l not in self.season.lefts for l in ls):
            return {
                "res": False,
                "reason": "wrongly_written_names_in_lefts",
                "detail": [c for c in ls if c not in self.season.lefts],
            }
        if any(r not in self.season.rights for r in rs):
            return {
                "res": False,
                "reason": "wrongly_written_names_in_rights",
                "detail": [c for c in rs if c not in self.season.rights],
            }

        return {"res": True, "reason": ""}

    def solution_possible(self, sol: Solution, end: int, includenight: bool) -> dict:

        if not super().solution_possible(sol, end, includenight)["res"]:
            return super().solution_possible(sol, end, includenight)

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
        self, sol: Solution, end: int, includenight: bool
    ) -> list[Solution]:
        if not self.solution_possible(sol, end, includenight)["res"]:
            return []


        if (
            len(sol) == self.season.num_matches
            and self.solution_possible(sol, end, includenight)["res"]
        ):
            return [sol]

        def zip_product(clefts, ordering) -> Solution:
            return Solution(p for p in zip(clefts, ordering))

        glefts, _, mmnum = get_candidates(sol)
        pos_matches = self.possible_matches_for_solution(sol, end, includenight)
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
        other_matches_list = list(
            map(lambda p: zip_product(pos_matches.keys(), p), products)
        )

        if mmnum > 0:
            isols = self.merge_mm_in_solution(sol, other_matches_list, end)
        else:
            isols = self.merge_mm_not_in_solution(
                sol, other_matches_list, end, includenight
            )
        # add self.sm as double_match
        solutions = [
            s | {(self.sm, r)}
            for r in smmatches
            for s in isols
            if self.solution_possible(s | {(self.sm, r)}, end, includenight)["res"]
        ] # todo: fix problem with too many mathes in night
        unique_sols = set()

        for sol in solutions:
            unique_sols.add(sol)

        return list(unique_sols)
