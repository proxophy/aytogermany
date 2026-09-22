from collections import Counter
import functools

from .ayto import Solver
from .models import *


class Normalo2024Solver(Solver):

    @functools.cache
    def no_match(self, l: str, r: str, end: int) -> bool:
        """Return whether a pair is known not to be a match up to `end`"""

        kpm = self.season.get_pms(end)
        hpm = [e for p in kpm for e in p]

        if (l, r) not in kpm and l in hpm:
            # lefts with multiple matches from matchboxes, appear x times
            # for x perfect matches
            mmls = [p for p in hpm if Counter(hpm)[p] > 1]
            # Normalo 24
            if len(mmls) < 3 and l in mmls and r not in hpm:
                # we only know two out of three of the multiple matches in Normalo 2024 at episode 10
                return False
            return True

        return super().no_match(l, r, end)

    def merge_mm_not_in_solution(
        self,
        sol: Solution,
        partial_solutions: list[Solution],
        end: int,
        include_night: bool,
    ) -> list[Solution]:
        # for documentation refer to Solver.merge_mm_not_in_solution
        
        sitting_nomatches = self.season.get_sitting_no_matches(sol, end, include_night)

        addable_lefts = {
            r: [
                l
                for l in self.season.lefts
                if not (self.no_match(l, r, end) or (l, r) in sitting_nomatches)
            ]
            for r in self.season.rights
        }

        solutions: list[Solution] = []

        for partial_solution in partial_solutions:
            base_matches = sol.union(partial_solution)

            sitting_lefts, sitting_rights = zip(*base_matches)
            mrs = [r for r in self.season.rights if r not in sitting_rights]

            if len(mrs) == 2:
                mr1, mr2 = mrs
                if self.season.mm in mrs:
                    # no multiple seatings yet , Mela not seated
                    for l in self.season.lefts:
                        if l in addable_lefts[mr1] and l in addable_lefts[mr2]:
                            solutions.append(base_matches | {(l, mr1), (l, mr2)})
                else:
                    # no mutiple seatings, Mela seated
                    dmleft = [l for (l, r) in base_matches if r == self.season.mm][
                        0
                    ]  # find partner of Mela
                    if dmleft in addable_lefts[mr1] and dmleft in addable_lefts[mr2]:
                        solutions.append(base_matches | {(dmleft, mr1), (dmleft, mr2)})

            elif len(mrs) == 1:
                counter = Counter(sitting_lefts)
                # left that already has two matches
                smleft = [l for l in self.season.lefts if counter[l] == 2][0]
                # missing right
                mr = [r for r in self.season.rights if r not in sitting_rights][0]
                if self.no_match(smleft, mr, end):
                    continue
                solutions.append(base_matches | {(smleft, mr)})

        return solutions
