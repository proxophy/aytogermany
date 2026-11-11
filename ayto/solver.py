from collections.abc import Generator, Iterable
from typing import override

from ayto.base import Solver

from .season import *

ALL_SEASONS = [
    "normalo2020",
    "normalo2021",
    "normalo2022",
    "normalo2023",
    "normalo2024",
    "normalo2025",
    "normalo2026",
    "vip2021",
    "vip2022",
    "vip2023",
    "vip2024",
    "vip2025",
    "vip2026",
]


def product_without_reps(
    arr: list[list[Person]],
) -> Generator[list[Person], Person, None]:
    used: set[Person] = set()
    current: list[Person] = []

    def rec(i: int) -> Generator[list[Person], Person, None]:
        if i == len(arr):
            yield current.copy()
            return

        for x in arr[i]:
            if x in used:
                continue

            used.add(x)
            current.append(x)

            yield from rec(i + 1)

            _ = current.pop()
            used.remove(x)

    yield from rec(0)


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
    Solver[Solution],
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
    def no_match_reason(
        self, l: Person, r: Person, episode: int, include_night: bool
    ) -> str | None:
        knowledge = self.knowledge_cache[(episode, include_night)]
        mb_no_matches = knowledge.mb_no_matches
        kpm = knowledge.perfect_matches
        pm_lefts = {a for a, _ in kpm}
        pm_rights = {b for _, b in kpm}
        mm_person = self.mm_person_cache[episode]

        # matchbox result was false
        if (l, r) in mb_no_matches:
            return "matchbox_false"

        # check for black out nights
        if (l, r) in knowledge.bp_no_matches:
            return "sitting_in_blackout_night"

        # one is part of known perfect match
        # consider multiple seated lefts
        # assumption: if dm is not known and you get double match,
        # you find the double match in the same episode
        if (l, r) not in kpm and (l in pm_lefts or r in pm_rights):
            # handle special case: for normalo2026, Alicia's match with Noel doesn't get revealed, despite Noel finding his match during the show
            if self.season.matchbox_reveals_dm:
                return "member_has_partner_already"
            else:
                # l cannot be part of multiple match
                if (
                    mm_person
                    and r != mm_person
                    and self.no_match(l, mm_person, episode, include_night)
                ):
                    return "l_cannot_have_double_match"
                elif episode < self.season.mm_known_from_week:
                    return "cannot_be_in_double_match"

        return None

    @override
    def find_format_violation(self, sol: Solution | set[Pair]) -> str | None:
        if len(sol) > self.season.num_matches:
            return "too_many_matches"

        if not self.season.double_match_for_right and has_dm_for_right(sol):
            return "double_match_for_right"

        return None

    @override
    def find_violation(
        self, sol: PairSet, episode: int, include_night: bool
    ) -> str | None:
        knowledge = self.knowledge_cache[(episode, include_night)]

        if (format_check := self.find_format_violation(sol)) is not None:
            return format_check

        # no known no matches
        for pair in sol:
            if self.no_match(*pair, episode, include_night):
                return "known_no_match"

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
                return "too_many_lights_in_night"
            elif sol_lights < night.lights and complete:
                return "not_enough_lights_in_night"

        return None

    @override
    def find_multiple_match_violation(self, sol: PairSet, episode: int) -> str | None:
        sol_dict = dict_rep(sol)
        # lefts with multiple partners
        mult_lefts = [l for l, rs in sol_dict.items() if len(rs) > 1]
        # double/triple match person
        mm_person = self.mm_person_cache[episode]

        if len(mult_lefts) >= 2:
            return "too_many_double_matches"

        elif len(mult_lefts) == 1:
            mult_left = mult_lefts[0]
            mult_rights = sol_dict[mult_left]
            if len(mult_rights) > self.season.max_mm_size:
                return "triple_match_not_allowed"

            if mm_person is not None:  # mm_person is not None
                _, seated_rights = zip(*sol)

                if mm_person not in sol_dict[mult_left]:
                    if len(mult_rights) == self.season.max_mm_size:
                        return "mm_not_in_complete_multiple_match"
                    elif mm_person in seated_rights:
                        # only used in normalo2024
                        return "multiple_match_not_possible_without_mm"

        return None

    @override
    def possible_matches_for_solution(
        self, sol: Solution, episode: int, include_night: bool
    ) -> dict[Person, list[Person]]:

        sitting_lefts, sitting_rights, _ = get_candidates(sol)

        knowledge = self.knowledge_cache[(episode, include_night)]
        sitting_no_matches = knowledge.get_sitting_no_matches(sol)

        mm_person = self.mm_person_cache[episode]
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

        mm_person = self.mm_person_cache[episode]
        knowledge = self.knowledge_cache[(episode, include_night)]
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
        # print(sum(len(x) for x in addable_lefts.values()))

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
