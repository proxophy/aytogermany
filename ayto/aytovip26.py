import itertools
from collections.abc import Iterable
from typing import override

from .ayto import Solver
from .models import *


class VIP2026Solver(Solver):
    sm: str | None = "Laurenz"
    double_match_for_right: bool = True

    @override
    def find_multiple_match_violation(
        self, sol: Solution, end: int
    ) -> str | None:

        supercheck = super().find_multiple_match_violation(sol, end)
        if supercheck is not None:
            return supercheck

        pdict: dict[str, list[str]] = {r: [] for r in self.season.rights}
        for l, r in sol:
            pdict[r].append(l)

        mutiplers = [
            r for r in self.season.rights if len(pdict[r]) > 1
        ]  # rights with multiple matches
        if len(mutiplers) > 1:
            return "more_than_one_double_right_match"
        elif len(mutiplers) == 1:
            multir = mutiplers[0]
            multils = pdict[multir]
            if len(multils) == 2 and self.sm not in multils:
                return "Laurenz_not_in_double_right_match"

        return None

    @override
    def generate_complete_solutions(
        self, sol: Solution, end: int, include_night: bool
    ) -> list[Solution]:
        if not self.solution_possible(sol, end, include_night):
            return []

        if len(sol) == self.season.num_matches and self.solution_possible(
            sol, end, include_night
        ):
            return [sol]

        def zip_product(clefts: Iterable[str], ordering: Iterable[str]) -> Solution:
            return Solution(p for p in zip(clefts, ordering))

        assert self.sm

        glefts, _, mmnum = get_candidates(sol)

        pos_matches: dict[str, list[str]] = self.possible_matches_for_solution(
            sol, end, include_night
        )
        _ = pos_matches.pop(self.sm, None)
        if self.sm in glefts:
            # remove pair with self.sm if necessary
            smmatches = [r for l, r in sol if l == self.sm]
            # print(smmatches)
            # isol = set(sol)
            sol = Solution(sol.difference({(self.sm, smmatches[0])}))
        else:
            smmatches = [
                r
                for r in self.season.rights
                if not self.no_match(self.sm, r, end) 
            ]

        products = [
            list(ps)
            for ps in itertools.product(*pos_matches.values())
            if len(set(ps)) == len(ps)
        ]
        partial_solutions = [zip_product(pos_matches.keys(), p) for p in products]

        if mmnum > 0:
            isols = self.merge_mm_in_solution(sol, partial_solutions)
        else:
            isols = self.merge_mm_not_in_solution(
                sol, partial_solutions, end, include_night
            )

        # add self.sm as double_match
        sitting_no_matches = self.season.get_sitting_no_matches(sol, end, include_night)
        solutions = [
            Solution(s | {(self.sm, r)})
            for r in smmatches
            for s in isols
            if (self.sm, r) not in sitting_no_matches
        ]

        return solutions
