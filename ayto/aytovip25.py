from .ayto import Solver, mm_left, dict_rep, get_candidates
from .models import *

import time
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
        s = time.time()
        sitting_nomatches: set[Pair] = set()
        nights = self.season.get_nights(end)
        if not includenight:
            nights = nights[:-1]
        for night in nights:
            pl = len(sol & night.pairs)
            if pl > night.lights:
                return []
            elif pl < night.lights:
                continue
            for p in night.pairs - sol:
                sitting_nomatches.add(p)

        addable_lefts = {
            r: [
                l
                for l in self.season.lefts
                if not (self.no_match(l, r, end) or (l, r) in sitting_nomatches)
            ]
            for r in self.season.rights
        }
        e = time.time()
        self.times["before_loop"] += e - s

        solutions: list[Solution] = []
        for othermatches in other_matches_list:
            s = time.time()
            tenplusmatches = sol.union(othermatches)
            base = tenplusmatches
            _, crights = zip(*tenplusmatches)
            missingrights = [r for r in self.season.rights if r not in crights]
            e = time.time()
            self.times["before_if"] += e - s

            if len(missingrights) == 2:
                s = time.time()
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
                        solutions += [
                            base | {(jimipartner, r1), (l2, r2)}
                            for l2 in addable_lefts[r2]
                            if l2 != jimipartner
                        ]
                    if jimipartner in addable_lefts[r2]:
                        solutions += [
                            base | {(jimipartner, r2), (l1, r1)}
                            for l1 in addable_lefts[r1]
                            if l1 != jimipartner
                        ]
                e = time.time()
                self.times["mr2"] += e - s
            elif len(missingrights) == 1:
                s = time.time()
                missing_right = missingrights[0]
                mml = mm_left(tenplusmatches)
                if missing_right == self.season.mm:
                    solutions += [
                        base | {(l, missing_right)}
                        for l in addable_lefts[missing_right]
                        if l != mml
                    ]
                else:
                    jimipartner = [
                        l for (l, r) in tenplusmatches if r == self.season.mm
                    ][0]
                    d = dict_rep(tenplusmatches)
                    # dont add to Jimi if he's already part of double match
                    if len(d[jimipartner]) == 2:
                        solutions += [
                            base | {(l, missing_right)}
                            for l in addable_lefts[missing_right]
                            if l != jimipartner
                        ]
                    else:
                        if jimipartner not in addable_lefts[missing_right]:
                            continue
                        solutions.append(base | {(jimipartner, missing_right)})
                e = time.time()
                self.times["mr1"] += e - s
        return solutions
