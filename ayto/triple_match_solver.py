from collections import Counter
from typing import override

from .season import *
from .solver import Solver


class TripleMatchSolver(Solver):
    @override
    def no_match_reason(
        self, l: Person, r: Person, end: int, include_night: bool
    ) -> str | None:
        """Return whether a pair is known not to be a match up to `end`"""

        kpm = self.season.get_pms(end, include_night)
        hpm = [e for p in kpm for e in p]

        counts = Counter(hpm)
        if (l, r) not in kpm and l in hpm:
            # lefts with multiple matches from matchboxes, appear x times
            # for x perfect matches
            mmls = [p for p in hpm if counts[p] > 1]
            # Normalo 24
            if len(mmls) < 3 and l in mmls and r not in hpm:
                # we only know two out of three of the multiple matches in Normalo 2024 at episode 10
                return None
            return "triple_match_already_exists"

        return super().no_match_reason(l, r, end, include_night)

    @override
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
                if not (
                    self.no_match(l, r, end, include_night)
                    or (l, r) in sitting_nomatches
                )
            ]
            for r in self.season.rights
        }

        solutions: list[Solution] = []
        mm_person = self.season.get_mm(end)

        for partial_solution in partial_solutions:
            base_matches = sol.union(partial_solution)

            sitting_lefts, sitting_rights = zip(*base_matches)
            mrs = [r for r in self.season.rights if r not in sitting_rights]

            if len(mrs) == 2:
                mr1, mr2 = mrs
                if mm_person in mrs:
                    # no multiple seatings yet , Mela not seated
                    for l in self.season.lefts:
                        if l in addable_lefts[mr1] and l in addable_lefts[mr2]:
                            solutions.append(base_matches | {(l, mr1), (l, mr2)})
                else:
                    # no mutiple seatings, Mela seated
                    dmleft = next(
                        l for (l, r) in base_matches if r == mm_person
                    )  # find partner of Mela
                    if dmleft in addable_lefts[mr1] and dmleft in addable_lefts[mr2]:
                        solutions.append(base_matches | {(dmleft, mr1), (dmleft, mr2)})

            elif len(mrs) == 1:
                counter = Counter(sitting_lefts)
                # left that already has two matches
                smleft = next(l for l in self.season.lefts if counter[l] == 2)
                # missing right
                mr = next(r for r in self.season.rights if r not in sitting_rights)
                if (
                    self.no_match(smleft, mr, end, include_night)
                    or (smleft, mr) in sitting_nomatches
                ):
                    continue
                solutions.append(base_matches | {(smleft, mr)})

        return solutions
