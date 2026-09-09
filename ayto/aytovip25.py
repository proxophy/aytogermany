from .ayto import AYTO
from .models import *

from typing import Optional, Union
from collections import Counter


class AYTOVIP2025(AYTO):

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
        self.two_dms = True
        # TODO: fix this

    def no_match(self, p: Pair, options: dict[str, bool]) -> bool:
        return super().no_match(p, options)

    def generate_partial_solutions(self, options: dict) -> list[Solution]:
        return super().generate_partial_solutions(options)

    def solution_possible(self, sol: Solution, options: dict) -> bool:
        if len(sol) == 0:
            return False
        p_lefts, p_rights = zip(*sol)
        r_counter = Counter(p_lefts)
        # no tripple matches
        if any([v > 2 for v in r_counter.values()]):
            return False
        return super().solution_possible(sol, options)

    def possible_matches_for_solution(
        self, sol: Solution, options: dict
    ) -> dict[str, list[str]]:
        return super().possible_matches_for_solution(sol, options)

    def merge_mm_in_solution(
        self, sol: Solution, other_matches_list: list[Solution], options: dict
    ):
        return super().merge_mm_in_solution(sol, other_matches_list, options)

    def merge_mm_not_in_solution(
        self, sol: Solution, other_matches_list: list[Solution], options: dict
    ):
        solutions = []
        addmatches_dict = {
            r: [
                Pair(l, r) for l in self.lefts if not self.no_match(Pair(l, r), options)
            ]
            for r in self.rights
        }
        for othermatches in other_matches_list:
            tenplusmatches = sol.union(othermatches)
            _, crights = zip(*tenplusmatches)
            missingrights = [r for r in self.rights if r not in crights]

            if len(missingrights) == 2:
                # add two matches
                r1, r2 = missingrights
                if r1 == self.dm or r2 == self.dm:
                    for l1, _ in addmatches_dict[r1]:
                        for l2, _ in addmatches_dict[r2]:
                            if l1 == l2:
                                continue
                            solutions.append(
                                tenplusmatches.union(
                                    Solution({Pair(l1, r1), Pair(l2, r2)})
                                )
                            )
                else:
                    dmleft = [l for (l, r) in tenplusmatches if r == self.dm][0]
                    if not self.no_match(Pair(dmleft, r1), options):
                        solutions += [
                            tenplusmatches.union(Solution({Pair(dmleft, r1), ap}))
                            for ap in addmatches_dict[r2]
                            if ap.l != dmleft
                        ]
                    if not self.no_match(Pair(dmleft, r2), options):
                        solutions += [
                            tenplusmatches.union(Solution({Pair(dmleft, r2), ap}))
                            for ap in addmatches_dict[r1]
                            if ap.l != dmleft
                        ]
            else:
                mr = missingrights[0]
                if mr == self.dm:
                    solutions += [
                        tenplusmatches.addpair(ap) for ap in addmatches_dict[mr]
                    ]
                else:
                    dmleft = [l for (l, r) in tenplusmatches if r == self.dm][0]
                    if (dmleft, mr) not in addmatches_dict[mr]:
                        continue
                    solutions.append(tenplusmatches.addpair(Pair(dmleft, mr)))
        return solutions

    # def get_solution_leftrights(self, sol: Solution):
    #     """Is the multiple match in the partial sol"""
    #     if len(sol) == 0:
    #         return set(), set(), False
    #     p_lefts, p_rights = zip(*sol)
    #     multiplels = {l for l in p_lefts if Counter(p_lefts)[l] == 2}
    #     return set(p_lefts), set(p_rights), len(multiplels) == 2

    def generate_complete_solutions(
        self, sol: Solution, options: dict
    ) -> list[Solution]:
        """Generating solutions"""
        return super().generate_complete_solutions(sol, options)


