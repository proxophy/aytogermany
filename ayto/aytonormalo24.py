from collections import Counter
import functools

from .ayto import Solver
from .models import *


class Normalo2024Solver(Solver):

    @functools.cache
    def no_match(self, l: str, r: str, end: int) -> bool:
        nomatch = super().no_match(l, r, end)

        kpm = self.season.get_pms(end)
        hpm = [e for p in kpm for e in p]

        if (l, r) not in kpm and l in hpm:
            # lefts with multiple matches from matchboxes, appear x times
            # for x perfect matches
            mmls = [p for p in hpm if Counter(hpm)[p] > 1]
            # Normalo 24
            if len(mmls) < 3 and l in mmls and r not in hpm:
                # Removed for simplicity
                # we only know two out of three of the multiple matches in Normalo 2024 at episode 10
                return False
            return True

        return nomatch

    def solution_possible(self, sol: Solution, end: int, includenight: bool) -> dict:
        pdict = {l: [] for l in self.season.lefts}
        sm_in_partialsol = False
        for l, r in sol:
            pdict[l].append(r)
            if r == self.season.mm:
                sm_in_partialsol = True

        # lefts with multiple matches
        multls = [l for l in self.season.lefts if len(pdict[l]) > 1]
        if len(multls) > 1:
            return {"res": False, "reason": f"more_than_one_multiple_match"}
        elif len(multls) == 1:
            multl = multls[0]
            multr = pdict[multl]
            # sm has to be one of multiple matches
            if sm_in_partialsol and (multl, self.season.mm) not in sol:  # type: ignore
                return {"res": False, "reason": "tripple_match_not_possiblee"}
            elif len(multr) == 3 and self.season.mm not in multr:
                return {"res": False, "reason": "Mela_not_in_tripple_match"}
        return super().solution_possible(sol, end, includenight)

    def possible_matches_for_solution(
        self, sol: Solution, end: int, includenight: bool
    ) -> dict[str, list[str]]:
        possible_matches = super().possible_matches_for_solution(sol, end, includenight)
        # Filter out sm
        possible_matches = {
            l: list(filter(lambda r: r != self.season.mm, possible_matches[l]))
            for l in possible_matches
        }
        return possible_matches

    def merge_mm_in_solution(
        self, sol: Solution, other_matches_list: list[Solution], end: int
    ) -> list[Solution]:
        if len(sol) > 0:
            g_lefts, g_rights = zip(*sol)
            g_lefts, g_rights = list(g_lefts), list(g_rights)
        else:
            g_lefts, g_rights = {}, {}

        not_sm_in_asm = self.season.mm is not None and self.season.mm not in g_rights

        if len(g_lefts) - len(set(g_lefts)) == 2:
            if self.season.mm not in g_rights:
                return []
            return [sol.union(othermatches) for othermatches in other_matches_list]

        elif len(g_lefts) - len(set(g_lefts)) == 1:
            counter = Counter(g_lefts)
            smleft = [l for l in self.season.lefts if counter[l] == 2][0]
            solutions = []
            for othermatches in other_matches_list:
                elevenmatches = sol.union(othermatches)
                _, crights = zip(*elevenmatches)
                missingright = [r for r in self.season.rights if r not in crights][0]
                # if we still have to add Mela: skip when somebody else is missing
                if not_sm_in_asm and missingright != self.season.mm:
                    continue
                elif self.no_match(smleft, missingright, end):
                    continue
                solutions.append(elevenmatches.addpair((smleft, missingright)))

            return solutions
        return []

    def merge_mm_not_in_solution(
        self,
        sol: Solution,
        other_matches_list: list[Solution],
        end: int,
        includenight: bool,
    ):
        solutions = []
        addmatches_dict = {
            r: [(l, r) for l in self.season.lefts if not self.no_match(l, r, end)]
            for r in self.season.rights
        }

        for othermatches in other_matches_list:
            tenmatches = sol.union(othermatches)

            clefts, crights = zip(*tenmatches)
            missingright = [r for r in self.season.rights if r not in crights][0]

            # Normalo 2024
            missingrights = [r for r in self.season.rights if r not in crights]
            if len(missingrights) == 2:
                mr1, mr2 = missingrights
                if self.season.mm in missingrights:
                    # no multiple seatings, Mela not seated
                    for l in self.season.lefts:
                        addmatches = Solution(frozenset(((l, mr1), (l, mr2))))
                        if all([not self.no_match(*p, end) for p in addmatches]):
                            solutions.append(tenmatches.union(addmatches))
                else:
                    # no mutiple seatings, Mela seated
                    dmleft = [l for (l, r) in tenmatches if r == self.season.mm][0]
                    addmatches = Solution(frozenset(((dmleft, mr1), (dmleft, mr2))))
                    if all([not self.no_match(*p, end) for p in addmatches]):
                        solutions.append(tenmatches.union(addmatches))
            elif len(missingrights) == 1:
                counter = Counter(clefts)
                if len([l for l in self.season.lefts if counter[l] == 2]) == 0:
                    print(tenmatches, missingrights, counter)
                    print(len(tenmatches))
                    assert False
                smleft = [l for l in self.season.lefts if counter[l] == 2][0]
                solutions = []
                not_sm_in_asm = self.season.mm is not None and self.season.mm not in crights
                for othermatches in other_matches_list:
                    elevenmatches = sol.union(othermatches)
                    _, crights = zip(*elevenmatches)
                    missingright = [r for r in self.season.rights if r not in crights][0]
                    # if we still have to add Mela: skip when somebody else is missing
                    if not_sm_in_asm and missingright != self.season.mm:
                        continue
                    elif self.no_match(smleft, missingright, end):
                        continue
                    solutions.append(elevenmatches.addpair((smleft, missingright)))


        return solutions
