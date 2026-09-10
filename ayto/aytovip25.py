from .ayto import Solver
from .models import *

from typing import Optional, Union
from collections import Counter


class VIP2025Solver(Solver):

    def solution_possible(self, sol: Solution, end: int) -> bool:
        if len(sol) == 0:
            return False
        p_lefts, p_rights = zip(*sol)
        r_counter = Counter(p_lefts)
        # no tripple matches
        if any([v > 2 for v in r_counter.values()]):
            return False
        return super().solution_possible(sol, end)

    def merge_mm_not_in_solution(
        self, sol: Solution, other_matches_list: list[Solution], end:int
    ):
        solutions = []
        addmatches_dict = {
            r: [
                Pair(l, r) for l in self.season.lefts if not self.no_match(Pair(l, r), end)
            ]
            for r in self.season.rights
        }
        for othermatches in other_matches_list:
            tenplusmatches = sol.union(othermatches)
            _, crights = zip(*tenplusmatches)
            missingrights = [r for r in self.season.rights if r not in crights]

            if len(missingrights) == 2:
                # add two matches
                r1, r2 = missingrights
                if r1 == self.season.dm or r2 == self.season.dm:
                    for l1, _ in addmatches_dict[r1]:
                        for l2, _ in addmatches_dict[r2]:
                            if l1 == l2:
                                continue
                            solutions.append(
                                tenplusmatches.union(
                                    Solution(frozenset({Pair(l1, r1), Pair(l2, r2)}))
                                )
                            )
                else:
                    dmleft = [l for (l, r) in tenplusmatches if r == self.season.dm][0]
                    if not self.no_match(Pair(dmleft, r1), end):
                        solutions += [
                            tenplusmatches.union(
                                Solution(frozenset({Pair(dmleft, r1), ap}))
                            )
                            for ap in addmatches_dict[r2]
                            if ap.l != dmleft
                        ]
                    if not self.no_match(Pair(dmleft, r2), end):
                        solutions += [
                            tenplusmatches.union(
                                Solution(frozenset({Pair(dmleft, r2), ap}))
                            )
                            for ap in addmatches_dict[r1]
                            if ap.l != dmleft
                        ]
            else:
                mr = missingrights[0]
                if mr == self.season.dm:
                    solutions += [
                        tenplusmatches.addpair(ap) for ap in addmatches_dict[mr]
                    ]
                else:
                    dmleft = [l for (l, r) in tenplusmatches if r == self.season.dm][0]
                    if (dmleft, mr) not in addmatches_dict[mr]:
                        continue
                    solutions.append(tenplusmatches.addpair(Pair(dmleft, mr)))
        return solutions

    