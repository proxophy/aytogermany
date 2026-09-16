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
        self,
        sol: Solution,
        other_matches_list: list[Solution],
        end: int,
        includenight: bool,
    ):
        solutions = []
        sitting_nomatches: dict[Pair, bool] = {}
        # Consider sitting matches as no matches if not in Solution and we have the right
        # amount of lights
        nights = self.season.get_nights(end)
        if not includenight:
            nights = nights[:-1]
        for night in nights:
            pl = sol.intersection_length(night.pairs)
            if pl > night.lights:
                return []
            elif pl < night.lights:
                continue
            for p in set(night.pairs) - set(sol.pairs):
                sitting_nomatches[p] = True
        addmatches_dict = {
            r: [
                (l, r)
                for l in self.season.lefts
                if not (
                    self.no_match(l, r, end) or sitting_nomatches.get((l, r), False)
                )
            ]
            for r in self.season.rights
        }
        # print("merge_mm_not_in_solution")
        for othermatches in other_matches_list:
            tenplusmatches = sol.union(othermatches)
            _, crights = zip(*tenplusmatches)
            missingrights = [r for r in self.season.rights if r not in crights]
            # print(missingrights)
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
                    jimipartner = [
                        l for (l, r) in tenplusmatches if r == self.season.mm
                    ][0]
                    # print(jimipartner)
                    if (jimipartner, r1) in addmatches_dict[r1]:
                        solutions += [
                            tenplusmatches.union(
                                Solution(frozenset({(jimipartner, r1), ap}))
                            )
                            for ap in addmatches_dict[r2]
                            if ap[0] != jimipartner
                        ]
                    if (jimipartner, r2) in addmatches_dict[r2]:
                        solutions += [
                            tenplusmatches.union(
                                Solution(frozenset({(jimipartner, r2), ap}))
                            )
                            for ap in addmatches_dict[r1]
                            if ap[0] != jimipartner
                        ]
            else:
                missing_right = missingrights[0]
                mm_left = tenplusmatches.mm_left()
                if missing_right == self.season.mm:
                    solutions += [
                        tenplusmatches.addpair(ap)
                        for ap in addmatches_dict[missing_right]
                        if ap[0] != mm_left
                    ]
                else:
                    jimipartner = [
                        l for (l, r) in tenplusmatches if r == self.season.mm
                    ][0]
                    d = tenplusmatches.dict_rep()
                    # dont add to Jimi if he's already part of double match
                    if len(d[jimipartner]) == 2:
                        solutions += [
                            tenplusmatches.addpair(ap)
                            for ap in addmatches_dict[missing_right]
                            if ap[0] != jimipartner
                        ]
                    else:
                        if (jimipartner, missing_right) not in addmatches_dict[
                            missing_right
                        ]:
                            continue
                        solutions.append(
                            tenplusmatches.addpair((jimipartner, missing_right))
                        )
        return solutions
