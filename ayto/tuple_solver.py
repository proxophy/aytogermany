from collections.abc import Iterable
from typing import override

from ayto.season import (
    Knowledge,
    Pair,
    PairSet,
    Person,
    Season,
    Solution,
    dict_rep,
    get_candidates,
    has_dm_for_right,
    has_two_dms,
)

from .base import NoMatchReason, Solver, Violation


def partial_sols(
    pos_matches: dict[Person, list[Person]],
) -> list[tuple[Solution, set[Person]]]:
    options = [[((l, r), r) for r in domain] for l, domain in pos_matches.items()]
    n = len(options)
    out: list[tuple[Solution, set[Person]]] = []

    def rec(i: int, used: set[Person], acc: frozenset[Pair]) -> None:
        if i == n:
            out.append((acc, used))
            return
        for pair, right in options[i]:
            if not (right in used):
                rec(i + 1, used | {right}, acc | {pair})

    rec(0, set(), frozenset())
    return out


class TupleSolver(
    Solver[Solution, set[Person]],
):
    """Solver for a typical AYTO season with one double match.
    Season-specific solvers inherit from this solver and can override its logic where necessary.
    Args:
        season (Season): Season object containing candidates, match boxes, matching nights and other information about the season
    """

    season: Season

    def __init__(self, season: Season) -> None:
        super().__init__(season)
        self.season = season

        self.knowledge_cache: dict[tuple[int, bool], Knowledge] = {}
        # todo: fix edge case
        for i in range(self.season.max_lights + 1):
            for b in [True, False]:
                self.knowledge_cache[(i, b)] = Knowledge(season, i, b)

    @override
    def from_pairs(self, pairs: Iterable[Pair]) -> Solution:
        return Solution(pairs)

    @override
    def to_pairs(self, sol: Solution) -> frozenset[Pair]:
        return sol

    @override
    def no_match_reason(
        self, l: Person, r: Person, episode: int, include_night: bool
    ) -> NoMatchReason | None:
        knowledge = self.knowledge_cache[(episode, include_night)]
        mb_no_matches = knowledge.mb_no_matches
        kpm = knowledge.perfect_matches
        pm_lefts = {a for a, _ in kpm}
        pm_rights = {b for _, b in kpm}
        mm_person = knowledge.mm

        # matchbox result was false
        if (l, r) in mb_no_matches:
            return NoMatchReason.MATCHBOX_FALSE

        # check for black out nights
        if (l, r) in knowledge.bp_no_matches:
            return NoMatchReason.SITTING_IN_BLACKOUT_NIGHT

        # one is part of known perfect match
        # consider multiple seated lefts
        # assumption: if dm is not known and you get double match,
        # you find the double match in the same episode
        if (l, r) not in kpm and (l in pm_lefts or r in pm_rights):
            # handle special case: for normalo2026, Alicia's match with Noel doesn't get revealed, despite Noel finding his match during the show
            if self.season.matchbox_reveals_dm:
                return NoMatchReason.MEMBER_HAS_PARTNER_ALREADY
            else:
                # l cannot be part of multiple match
                if (
                    mm_person
                    and r != mm_person
                    and self.no_match(l, mm_person, episode, include_night)
                ):
                    return NoMatchReason.L_CANNOT_HAVE_SECOND_MATCH  # todo: rename this
                elif episode < self.season.mm_known_from_week:
                    return NoMatchReason.L_CANNOT_HAVE_SECOND_MATCH

        return None

    @override
    def find_format_violation(self, sol: Solution | set[Pair]) -> Violation | None:
        if len(sol) > self.season.num_matches:
            return Violation.TOO_MANY_MATCHES

        if not self.season.double_match_for_right and has_dm_for_right(sol):
            return Violation.DOUBLE_MATCH_FOR_RIGHT

        return None

    @override
    def find_violation(
        self, sol: PairSet, episode: int, include_night: bool
    ) -> Violation | None:
        knowledge = self.knowledge_cache[(episode, include_night)]

        if (format_check := self.find_format_violation(sol)) is not None:
            return format_check

        # no known no matches
        for pair in sol:
            if self.no_match(*pair, episode, include_night):
                return Violation.CONTAINS_KNOWN_NO_MATCH

        complete: bool = len(sol) == self.season.num_matches

        # check conditions for double matches
        double_match_check = self.find_multiple_match_violation(sol, episode)
        if double_match_check is not None:
            return double_match_check

        # number of lights with pairs matching lights in nights
        # if we don't have num_matches pairs, we allow lesser lights
        nights = knowledge.nights

        for _, night in enumerate(nights):
            sol_lights: int = len(sol & night.pairs)

            if sol_lights > night.lights:
                return Violation.TOO_MANY_LIGHTS_IN_NIGHT
            elif sol_lights < night.lights and complete:
                return Violation.NOT_ENOUGH_LIGHTS_IN_NIGHT

        return None

    @override
    def find_multiple_match_violation(
        self, sol: PairSet, episode: int
    ) -> Violation | None:
        sol_dict = dict_rep(sol)
        # lefts with multiple partners
        mult_lefts = [l for l, rs in sol_dict.items() if len(rs) > 1]
        # double/triple match person
        knowledge = self.knowledge_cache[(episode, True)]
        mm_person = knowledge.mm

        if not self.season.double_match_for_right and has_dm_for_right(sol):
            return Violation.DOUBLE_MATCH_FOR_RIGHT

        if len(mult_lefts) >= 2:
            return Violation.TOO_MANY_MULTIPLE_MATCHES

        elif len(mult_lefts) == 1:
            mult_left = mult_lefts[0]
            mult_rights = sol_dict[mult_left]
            if len(mult_rights) > self.season.max_mm_size:
                return Violation.MULTIPLE_MATCH_TOO_BIG

            if mm_person is not None:  # mm_person is not None
                _, seated_rights = zip(*sol)

                # assumes there is one multiple match, case of two double matches overrides this
                if mm_person not in sol_dict[mult_left] and (
                    mm_person in seated_rights
                    or len(mult_rights) == self.season.max_mm_size
                ):
                    return Violation.MM_NOT_IN_MULTIPLE_MATCH

        return None

    @override
    def possible_matches_for_solution(
        self, sol: Solution, episode: int, include_night: bool
    ) -> dict[Person, list[Person]]:

        sitting_lefts, sitting_rights, _ = get_candidates(sol)

        knowledge = self.knowledge_cache[(episode, include_night)]
        sitting_no_matches = knowledge.get_sitting_no_matches(sol)

        mm_person = knowledge.mm
        pos_matches = {
            l: [
                r
                for r in self.season.rights
                if not self.no_match(l, r, episode, include_night)
                and not (l, r) in sitting_no_matches
                and r != mm_person
                and r not in sitting_rights
            ]
            for l in self.season.lefts
            if l not in sitting_lefts
        }
        return pos_matches

    @override
    def merge_mm_in_solution(
        self, sol: Solution, partial_solutions: list[tuple[Solution, set[int]]]
    ) -> list[Solution]:
        _, _, mm_num = get_candidates(sol)
        if mm_num == 0:
            raise ValueError(
                "merge_mm_in_solution shouldn't be called if multiple match(es) are not in Solution"
            )

        sols = [sol | partial_solution for partial_solution, _ in partial_solutions]
        return sols

    @override
    def merge_mm_not_in_solution(
        self,
        sol: Solution,
        partial_solutions: list[tuple[Solution, set[int]]],
        episode: int,
        include_night: bool,
    ) -> list[Solution]:

        knowledge = self.knowledge_cache[(episode, include_night)]
        mm_person = knowledge.mm
        sitting_no_matches = knowledge.get_sitting_no_matches(sol)

        # compute possible partners left for missing rights
        addable_lefts = {
            r: [
                l
                for l in self.season.lefts
                if not (
                    self.no_match(l, r, episode, include_night)
                    or (l, r) in sitting_no_matches
                    or (
                        self.season.double_match_for_right
                        and l == self.season.extra_left
                    )
                )
            ]
            for r in self.season.rights
        }

        solutions: list[Solution] = []
        full = set(self.season.rights)
        _, sitting_rights, _ = get_candidates(sol)

        for partial, used in partial_solutions:
            base_matches = sol | partial
            free = full - used - sitting_rights

            # right that is missing in solution
            mr = free.pop()
            if mm_person is not None:
                # mm_person is not None
                if mr == mm_person:
                    solutions.extend(
                        Solution(base_matches | {(l, mr)}) for l in addable_lefts[mr]
                    )
                else:
                    dm_left = next(
                        l for (l, r) in base_matches if r == mm_person
                    )  # left that matches with mm
                    if dm_left not in addable_lefts[mr]:
                        continue
                    solutions.append(Solution(base_matches | {(dm_left, mr)}))
            else:
                # Normalo 2023/24/25/26: dm not known, match mr with everyone possible
                solutions.extend(
                    [Solution(base_matches | {(l, mr)}) for l in addable_lefts[mr]]
                )

        return solutions

    @override
    def generate_complete_solutions(
        self, sol: Solution, episode: int, include_night: bool
    ) -> list[Solution]:

        if not self.solution_possible(sol, episode, include_night):
            return []
        if len(sol) == self.season.num_matches:
            return [sol]

        _, _, mm_num = get_candidates(sol)
        pos_matches: dict[Person, list[Person]] = self.possible_matches_for_solution(
            sol, episode, include_night
        )

        # build pairs for lefts that are not in the given solution
        partial_solutions = partial_sols(pos_matches)

        if (self.season.two_dms and has_two_dms(sol)) or (
            not self.season.two_dms and mm_num == self.season.max_mm_size
        ):
            # Multiple match is already in Solution
            sols = self.merge_mm_in_solution(sol, partial_solutions)
            return sols

        solutions = self.merge_mm_not_in_solution(
            sol, partial_solutions, episode, include_night
        )

        solutions = list(set(solutions))

        return solutions
