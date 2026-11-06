import functools
import itertools
from collections.abc import Generator, Iterable

from ayto.season import Person

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


class Solver:
    """Solver for a typical AYTO season with one double match.
    Season-specific solvers inhert from this solver and can override its logic where necessary.
    Args:
        season (Season): Season object containing candidates, match boxes, matching nights and other information about the season
    """

    season: Season

    def __init__(self, season: Season) -> None:
        self.season = season
        self._no_match_cache: dict[tuple[Person, Person, int, bool], bool] = {}
        
        self.knowledge_cache: dict[tuple[int, bool], Knowledge] = {} # todo: fix edge case
        for i in range(self.season.max_lights + 1):
            for b in [True, False]:
                self.knowledge_cache[(i, b)] = Knowledge(season, i, b)
        self.mm_person_cache: list[Person|None] = [season.get_mm(ep) for ep in range(self.season.max_lights + 1)]


    def no_match(self, l: Person, r: Person, episode: int, include_night: bool) -> bool:
        """Return whether `(l, r)` is known not to be a match up to a given episode"""
        key = (l, r, episode, include_night)
        cached = self._no_match_cache.get(key)
        if cached is None:
            cached = self.no_match_reason(l, r, episode, include_night) is not None
            self._no_match_cache[key] = cached
        return cached

    def no_match_reason(
        self, l: Person, r: Person, episode: int, include_night: bool
    ) -> str | None:
        """Return the reason why `(l, r)` is known not to be a match up to a given episode"""
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
        if (l,r) in knowledge.bp_no_matches:
            return "sitting_in_blackout_night"

        # one is part of known perfect match
        # consider multiple seated lefts
        # assumption: if dm is not known and you get double match,
        # you find the double match in the same episode
        if (l, r) not in kpm and (l in pm_lefts or r in pm_rights):
            # handle special case: for normalo2026, Alicia's match with Noel doesnt get revealed, despite Noel finding his match during the show
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

    def find_format_violation(self, sol: Solution | set[Pair]) -> str | None:
        """Return the first format violation found in the solution, if any" """

        if len(sol) > self.season.num_matches:
            return "too_many_matches"

        if not self.season.double_match_for_right and has_dm_for_right(sol):
            return "double_match_for_right"

        for l, r in sol:
            if l not in self.season.lefts:
                return "wrongly_written_names_as_left"
            elif r not in self.season.rights:
                return "wrongly_written_names_as_right"
        return None

    def solution_possible(self, sol: Solution, episode: int, include_night: bool) -> bool:
        """Check whether the solution is feasible up to a given episode

        Args:
            sol: Solution containing the pairs to cneck
            episode: Last episode to consider
            include_night: Whether the matching night of the last episode should be considered

        Returns:
            `True` if no violation was found, ``False`` otherwise.
        """
        return self.find_violation(sol, episode, include_night) is None

    def find_violation(
        self, sol: SolutionType, episode: int, include_night: bool
    ) -> str | None:
        """Return the first logic violation found in the solution, if any"""
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
        # if we don't have self.nummatches pairs, we allow lesser lights
        nights = knowledge.nights

        for _, night in enumerate(nights):
            sol_lights: int = len(sol & night.pairs)

            if sol_lights > night.lights:
                return "too_many_lights_in_night"
            elif sol_lights < night.lights and complete:
                return "not_enough_lights_in_night"

        return None

    def find_multiple_match_violation(self, sol: SolutionType, episode: int) -> str | None:
        """Return the first multiple-match violation found in the solution, if any"""
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
                    elif  mm_person in seated_rights:
                    # only used in normalo2024
                        return "multiple_match_not_possible_without_mm"

        return None

    def combine_possible_solutions(
        self,
        solutions1: list[Solution],
        solutions2: list[Solution],
        episode: int,
        include_night: bool,
    ) -> list[Solution]:
        """Combines solutions from two lists and keeps feasible combinations, given `episode`(inclusive) and `include_night`"""

        m_asm: set[Solution] = set()
        for s1, s2 in itertools.product(solutions1, solutions2):
            s3 = Solution(s1 | s2)
            if self.solution_possible(s3, episode, include_night):
                m_asm.add(s3)

        return list(m_asm)

    def possible_matches_for_solution(
        self, sol: Solution, episode: int, include_night: bool
    ) -> dict[Person, list[Person]]:
        """Given a solution, return a dict for the possible matches for the lefts that are not in the solution, given `episode`(inclusive) and `includenight`"""

        sitting_lefts, sitting_rights, _ = get_candidates(sol)

        knowledge = self.knowledge_cache[(episode, include_night)]
        sitting_nomatches = knowledge.get_sitting_no_matches(sol)

        mm_person = self.mm_person_cache[episode]
        pos_matches = {
            l: [
                r
                for r in self.season.rights
                if not self.no_match(l, r, episode, include_night)
                and not (l, r) in sitting_nomatches
                and r != mm_person
                and r not in sitting_rights
            ]
            for l in self.season.lefts
            if l not in sitting_lefts
        }
        return pos_matches

    def merge_mm_in_solution(
        self, sol: Solution, partial_solutions: list[Solution]
    ) -> list[Solution]:
        """Merge solution with partial solutions to complete solution when the multiple match(es) are present the solution, given `episode`(inclusive)"""

        _, _, mm_num = get_candidates(sol)
        if mm_num == 0:
            raise ValueError(
                "merge_mm_in_solution shouldn't be called if muliple match(es) are not in Solution"
            )

        sols = [
            sol | partial_solution
            for partial_solution in partial_solutions
        ]
        return sols

    def merge_mm_not_in_solution(
        self,
        sol: Solution,
        partial_solutions: list[Solution],
        episode: int,
        include_night: bool,
    ) -> list[Solution]:
        """Merge solution with partial solutions to complete solutions when the multiple match(es) are not present the solution, given `episode`(inclusive) and `includenight`"""

        mm_person = self.mm_person_cache[episode]
        knowledge = self.knowledge_cache[(episode, include_night)]
        sitting_nomatches = knowledge.get_sitting_no_matches(sol)

        # compute possible partners left for missing rights
        addable_lefts = {
            r: [
                l
                for l in self.season.lefts
                if not (
                    self.no_match(l, r, episode, include_night)
                    or (l, r) in sitting_nomatches
                    or (self.season.extra_left_name and l == self.season.extra_left)
                )
            ]
            for r in self.season.rights
        }
        # print(sum(len(x) for x in addable_lefts.values()))

        solutions: list[Solution] = []

        for partial_solution in partial_solutions:
            base_matches = sol | partial_solution

            _, sitting_rights, _ = get_candidates(base_matches)
            # right that is missing in solution
            mr = next(r for r in self.season.rights if r not in sitting_rights)
            if mm_person is not None:
                # mm_person is not None
                if mr == mm_person:
                    solutions.extend(
                        Solution(base_matches | {(l, mr)}) for l in addable_lefts[mr]
                    )
                else:
                    dmleft = next(
                        l for (l, r) in base_matches if r == mm_person
                    )  # left that matches with mm
                    if dmleft not in addable_lefts[mr]:
                        continue
                    solutions.append(Solution(base_matches | {(dmleft, mr)}))
            else:
                # Normalo 2023/24/25/26: dm not known, match mr with everyone possible
                solutions.extend(
                    [Solution(base_matches | {(l, mr)}) for l in addable_lefts[mr]]
                )

        return solutions

    def generate_complete_solutions(
        self, sol: Solution, episode: int, include_night: bool
    ) -> list[Solution]:
        """Generate complete solutions by extending a partial solution.

        Args:
            sol (Solution): Partial solution to extend.
            episode (int): Last episode to consider, inclusive.
            include_night (bool): Whether to consider the matching night of
                the last episode.

        Returns:
            list[Solution]: Complete solutions consistent with the given partial
                solution and the known constraints.
        """

        if not self.solution_possible(sol, episode, include_night):
            return []
        if len(sol) == self.season.num_matches:
            return [sol]

        def zip_product(
            clefts: Iterable[Person], ordering: Iterable[Person]
        ) -> Solution:
            return Solution(p for p in zip(clefts, ordering))

        _, _, mm_num = get_candidates(sol)
        pos_matches: dict[Person, list[Person]] = self.possible_matches_for_solution(sol, episode, include_night)

        # build pairs for lefts that are not in the given solution
        products = [list(ps) for ps in product_without_reps(list(pos_matches.values()))]
        partial_solutions = [zip_product(pos_matches.keys(), p) for p in products]

        if (self.season.two_dms and has_two_dms(sol)) or (
            not self.season.two_dms and mm_num == self.season.max_mm_size
        ):
            # Multiple match is already in Solution
            sols = self.merge_mm_in_solution(sol, partial_solutions)
            return sols

        solutions = self.merge_mm_not_in_solution(
            sol, partial_solutions, episode, include_night
        )

        return solutions

    def generate_partial_solutions(
        self, episode: int, include_night: bool
    ) -> list[Solution]:
        """Generate partial solutions from the matching night.

        For each matching night, the required number of matching pairs is selected and the resulting partial solutions are combined.
        Args:
            episode (int): Last episode to consider, inclusive.
            include_night (bool): Whether to consider the matching night of
                the last episode.

        Returns:
            list[Solution]: Partial solutions consistent with the matching-night
                results and known constraints.
        """

        knowledge = self.knowledge_cache[(episode, include_night)]
        nights = knowledge.nights
        kpm = knowledge.perfect_matches

        if len(nights) == 0:
            return [Solution(kpm)]

        pairs_per_night: list[list[Solution]] = []
        for i, night in enumerate(nights):
            invalid_pairs = list(
                filter(
                    lambda p: self.no_match(*p, episode, include_night) or p in kpm,
                    night.pairs,
                )
            )
            definite_matches = list(filter(lambda p: p in kpm, night.pairs))
            remaining = set(night.pairs) - set(definite_matches) - set(invalid_pairs)

            missinglights = night.lights - len(definite_matches)
            if missinglights < 0:
                raise ValueError(
                    f"{self.season.name}, night {i}: {len(definite_matches)} known perfect  \
                    matches are seated but the night only has {night.lights} lights"
                )
            combinations = [
                Solution(set(comb).union(kpm))
                for comb in itertools.combinations(remaining, missinglights)
            ]

            pairs_per_night.append(combinations)

        merged_solutions: list[Solution] = functools.reduce(
            lambda s1, s2: self.combine_possible_solutions(s1, s2, episode, include_night),
            pairs_per_night,
        )

        return merged_solutions

    def solve(
        self, episode: int, include_night: bool = True, validate: bool = False
    ) -> list[Solution]:
        """Solve season and return all possible solutions

        Args:
            episode (int): Last episode to consider
            include_night (bool): Wheter the matching night of the last episode should be considered.Defaults to True.
            validate (bool, optional): Whether solutions should be vaidated by solution_possible after generating them. Defaults to False.

        Returns:
            list[Solution]: list of all possible solutions
        """

        # reduction of the search space by generating partial solutions from nights
        partial_solutions = self.generate_partial_solutions(episode, include_night)

        solutions_unfiltered: set[Solution] = set()
        for g in partial_solutions:
            sols_g = self.generate_complete_solutions(g, episode, include_night)
            solutions_unfiltered.update(sols_g)

        # the tests should guarantee that all generated solutions are possible
        if validate:
            solutions = [
                s
                for s in solutions_unfiltered
                if self.solution_possible(s, episode, include_night)
            ]
        else:
            solutions = list(solutions_unfiltered)

        return list(solutions)
