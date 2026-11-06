from collections import Counter
from typing import override

from .season import *
from .solver import Solver


class TripleMatchSolver(Solver):
    @override
    def no_match_reason(
        self, l: Person, r: Person, episode: int, include_night: bool
    ) -> str | None:
        """Return whether a pair is known not to be a match up to `end`"""
        knowledge = self.knowledge_cache[(episode, include_night)]
        kpm = knowledge.perfect_matches
        pm_lefts = [a for a, _ in kpm]
        pm_rights = {b for _, b in kpm}

        counts = Counter(pm_lefts)
        if (l, r) not in kpm and l in pm_lefts:
            # lefts with multiple matches from matchboxes, appear x times
            # for x perfect matches
            mmls = [p for p in pm_lefts if counts[p] > 1]
            # Normalo 24
            if len(mmls) < 3 and l in mmls and r not in pm_rights:
                # we only know two out of three of the multiple matches in Normalo 2024 at episode 10
                return None
            return "triple_match_already_exists"

        return super().no_match_reason(l, r, episode, include_night)

    @override
    def merge_mm_not_in_solution(
        self,
        sol: Solution,
        partial_solutions: list[Solution],
        episode: int,
        include_night: bool,
    ) -> list[Solution]:
        # for documentation refer to Solver.merge_mm_not_in_solution
        knowledge = self.knowledge_cache[(episode, include_night)]
        sitting_nomatches = knowledge.get_sitting_no_matches(sol)

        addable_lefts = {
            r: [
                l
                for l in self.season.lefts
                if not (
                    self.no_match(l, r, episode, include_night)
                    or (l, r) in sitting_nomatches
                )
            ]
            for r in self.season.rights
        }

        solutions: list[Solution] = []
        mm_person = self.mm_person_cache[episode]

        for partial_solution in partial_solutions:
            base_matches = sol.union(partial_solution)

            sitting_lefts, sitting_rights= zip(*base_matches)
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
                    self.no_match(smleft, mr, episode, include_night)
                    or (smleft, mr) in sitting_nomatches
                ):
                    continue
                solutions.append(base_matches | {(smleft, mr)})

        return solutions
