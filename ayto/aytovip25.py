from collections import Counter

from .ayto import Solver, mm_left, dict_rep, get_candidates
from .models import *


class VIP2025Solver(Solver):

    def check_multiple_match_logic(self, sol: Solution, end: int) -> dict:
        # VIP2025 allows two double matches, one of those must include Jimi

        seated_lefts, seated_right = zip(*sol)
        sol_dict = dict_rep(sol)

        mult_lefts = [
            l for l, rs in sol_dict.items() if len(rs) > 1
        ]  # lefts with multiple partners
        mult_rights = [
            e for l in mult_lefts for e in sol_dict[l]
        ]  # rights part of multi match
        mm = self.season.mm if end >= self.season.mm_known_after_week else None

        # if Jimi is seated and we have two double matches, he must be part of one of them
        if mm in seated_right and len(mult_lefts) == 2 and mm not in mult_rights:
            return {"res": False, "reason": "Jimi_not_in_double_match"}

        r_counter = Counter(seated_lefts)
        # no tripple matches
        if any([v > 2 for v in r_counter.values()]):
            return {"res": False, "reason": f"no_tripple_matches_allowed"}
        return {"res": True, "reason": ""}

    def merge_mm_not_in_solution(
        self,
        sol: Solution,
        partial_solutions: list[Solution],
        end: int,
        include_night: bool,
    ) -> list[Solution]:
        sitting_nomatches = self.season.get_sitting_no_matches(sol, end, include_night)

        addable_lefts = {
            r: [
                l
                for l in self.season.lefts
                if not (self.no_match(l, r, end) or (l, r) in sitting_nomatches)
            ]
            for r in self.season.rights
        }

        solutions: list[Solution] = list()

        for partial_solution in partial_solutions:

            tenplusmatches = sol.union(partial_solution)
            base = tenplusmatches
            _, crights = zip(*tenplusmatches)
            missingrights = [r for r in self.season.rights if r not in crights]

            if len(missingrights) == 2:
                # add two matches
                r1, r2 = missingrights
                if r1 == self.season.mm or r2 == self.season.mm:
                    for l1 in addable_lefts[r1]:
                        for l2 in addable_lefts[r2]:
                            if l1 == l2:
                                continue

                            x = base | {(l1, r1), (l2, r2)}
                            solutions.append(x)
                else:
                    jimipartner = [
                        l for (l, r) in tenplusmatches if r == self.season.mm
                    ][0]
                    # print(jimipartner)
                    if jimipartner in addable_lefts[r1]:
                        solutions.extend(
                            base | {(jimipartner, r1), (l2, r2)}
                            for l2 in addable_lefts[r2]
                            if l2 != jimipartner
                        )
                    if jimipartner in addable_lefts[r2]:
                        solutions.extend(
                            base | {(jimipartner, r2), (l1, r1)}
                            for l1 in addable_lefts[r1]
                            if l1 != jimipartner
                        )
            elif len(missingrights) == 1:
                mr = missingrights[0]
                mml = mm_left(tenplusmatches)
                if mr == self.season.mm:
                    solutions.extend(
                        base | {(l, mr)} for l in addable_lefts[mr] if l != mml
                    )
                else:
                    jimipartner = [
                        l for (l, r) in tenplusmatches if r == self.season.mm
                    ][0]
                    d = dict_rep(tenplusmatches)
                    # dont add to Jimi if he's already part of double match
                    if len(d[jimipartner]) == 2:
                        solutions.extend(
                            base | {(l, mr)}
                            for l in addable_lefts[mr]
                            if l != jimipartner
                        )
                    else:
                        if jimipartner not in addable_lefts[mr]:
                            continue
                        solutions.append(base | {(jimipartner, mr)})
        return solutions
