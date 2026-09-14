from .ayto import Solver
from .models import *

from collections import Counter


class VIP2025Solver(Solver):

    def solution_possible(self, sol: Solution, end: int, includenight: bool) -> dict:
        p_lefts, p_rights = zip(*sol)
        r_counter = Counter(p_lefts)
        # no tripple matches
        if any([v > 2 for v in r_counter.values()]):
            return {"res": False, "reason": f"no_tripple_matches_allowed"}
        return super().solution_possible(sol, end, includenight)


    def merge_mm_not_in_solution(
        self, sol: Solution, other_matches_list: list[Solution], end:int, includenight
    ):
        solutions = []
        addmatches_dict = {
            r: [
                (l, r) for l in self.season.lefts if not self.no_match(l, r, end)
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
                if r1 == self.season.mm or r2 == self.season.mm:
                    for l1, _ in addmatches_dict[r1]:
                        for l2, _ in addmatches_dict[r2]:
                            if l1 == l2:
                                continue
                            solutions.append(
                                tenplusmatches.union(
                                    Solution(frozenset({(l1, r1), (l2, r2)}))
                                )
                            )
                else:
                    dmleft = [l for (l, r) in tenplusmatches if r == self.season.mm][0]
                    if not self.no_match(dmleft, r1, end):
                        solutions += [
                            tenplusmatches.union(
                                Solution(frozenset({(dmleft, r1), ap}))
                            )
                            for ap in addmatches_dict[r2]
                            if ap[0] != dmleft
                        ]
                    if not self.no_match(dmleft, r2, end):
                        solutions += [
                            tenplusmatches.union(
                                Solution(frozenset({(dmleft, r2), ap}))
                            )
                            for ap in addmatches_dict[r1]
                            if ap[0] != dmleft
                        ]
            else:
                mr = missingrights[0]
                if mr == self.season.mm:
                    solutions += [
                        tenplusmatches.addpair(ap) for ap in addmatches_dict[mr]
                    ]
                else:
                    dmleft = [l for (l, r) in tenplusmatches if r == self.season.mm][0]
                    if (dmleft, mr) not in addmatches_dict[mr]:
                        continue
                    solutions.append(tenplusmatches.addpair((dmleft, mr)))
        return solutions

    