import functools
import itertools
from collections.abc import Generator, Iterable

from .models import *

allseasons = [
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


def product_without_reps(arr: list[list[str]]) -> Generator[list[str], str, None]:
    used: set[str] = set()
    current: list[str] = []

    def rec(i: int) -> Generator[list[str], str, None]:
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
    sm: str | None = None
    double_match_for_right: bool = False

    def __init__(self, season: Season) -> None:
        self.season = season
        self.bo_nights: set[int] = self.season.get_black_out_nights()
       
    @functools.lru_cache  # noqa: B019
    def no_match(self, l: str, r: str, end: int) -> bool:
        """Return whether `(l, r)` is known not to be a match up to a given episode"""
        return self.no_match_reason(l, r, end) is not None

    def no_match_reason(self, l: str, r: str, end: int) -> str | None:
        """Return the reason why `(l, r)` is known not to be a match up to a given episode"""
        end = max(0, min(end, 10))

        mb = self.season.get_matchboxes(end)
        kpm = self.season.get_pms(end)
        hpm = [e for pp in kpm for e in pp]

        # matchbox result was false
        if (l, r) in mb and not mb[(l, r)]:
            return "matchbox_false"

        # check for black out nights
        for bo in self.bo_nights:
            if bo > end:
                break
            if (l, r) in self.season.nights[bo].pairs and (l, r) not in kpm:
                return "sitting_in_blackout_night"

        # one is part of known perfect match
        # consider multiple seated lefts
        # assumption: if dm is not known and you get double match,
        # you find the double match in the same episode
        if (l, r) not in kpm and (l in hpm or r in hpm):
            # handle special case: for normalo2026, Alicia's match with Noel doesnt get revealed, despite Noel finding his match during the show
            if self.season.mb_reveals_dm:
                return "member_has_partner_already"
            else:
                mm = self.season.get_mm(end)
                # l cannot be part of multiple match
                if mm and r != mm and self.no_match(l, mm, end):
                    return "l_cannot_have_double_match"
                elif end < self.season.mm_known_after_week:
                    return "cannot_be_in_double_match"

        return None

    def find_format_violation(self, sol: Solution | set[Pair]) -> str | None:
        """Return the first fomrat violation found in the solution, if any" """

        if len(sol) > self.season.num_matches:
            return "too_many_matches"

        if not self.double_match_for_right and double_match_for_right(sol):
            return "double_match_for_right"

        for l, r in sol:
            if l not in self.season.lefts:
                return "wrongly_written_names_as_left"
            elif r not in self.season.rights:
                return "wrongly_written_names_as_right"
        return None

    def solution_possible(
        self, sol: Solution | set[Pair], end: int, include_night: bool
    ) -> bool:
        """Check whether the solution is feasible up to a given episode

        Args:
            sol: Solution containing the pairs to cneck
            end: Last episode to consider
            include_night: Whether the matching night of the last episode should be considered

        Returns:
            `True` if no violation was found, ``False`` otherwise.
        """
        return self.find_violation(sol, end, include_night) is None

    def find_violation(
        self, sol: Solution | set[Pair], end: int, include_night: bool
    ) -> str | None:
        """Return the first logic violation found in the solution, if any"""

        if len(sol) > self.season.num_matches:
            return "too_many_matches"

        format_check = self.find_format_violation(sol)
        if format_check is not None:
            return format_check

        end = max(0, min(end, 10))

        # no known no matches
        for pair in sol:
            if self.no_match(*pair, end):
                return "known_no_match"

        complete: bool = len(sol) == self.season.num_matches

        # check conditions for double matches
        double_match_check = self.find_multiple_match_violation(sol, end)
        if double_match_check is not None:
            return double_match_check

        # number of lights with pairs matching lights in nights
        # if we don't have self.nummatches pairs, we allow lesser lights
        nights = self.season.get_nights(end)
        if not include_night:
            nights = nights[:-1]

        for _, night in enumerate(nights):
            sol_lights: int = len(sol & night.pairs)

            if sol_lights > night.lights:
                return "too_many_lights_in_night"
            elif sol_lights < night.lights and complete:
                return "not_enough_lights_in_night"

        return None

    def find_multiple_match_violation(
        self, sol: Solution | set[Pair], end: int
    ) -> str | None:
        """Return the first multiple-match violation found in the solution, if any"""

        sol_dict = dict_rep(sol)
        # lefts with multiple partners
        mult_lefts = [l for l, rs in sol_dict.items() if len(rs) > 1]
        # double/tripple match person
        mm = self.season.get_mm(end)

        if len(mult_lefts) >= 2:
            return "too_many_double_matches"

        elif len(mult_lefts) == 1:
            mult_left = mult_lefts[0]
            mult_rights = sol_dict[mult_left]
            if len(mult_rights) > self.season.max_mm_size:
                return "triple_match_not_allowed"

            if mm:  # mm is not None
                _, seated_rights = zip(*sol)

                if (
                    len(mult_rights) == self.season.max_mm_size
                    and mm not in sol_dict[mult_left]
                ):
                    return "mm_not_in_complete_multiple_match"
                elif mm in seated_rights and mm not in sol_dict[mult_left]:
                    # only used in normalo2024
                    return "multiple_match_not_possible_without_mm"

        return None

    def combine_possible_solutions(
        self,
        solutions1: list[Solution],
        solutions2: list[Solution],
        end: int,
        include_night: bool,
    ) -> list[Solution]:
        """Combines solutions from two lists and keeps feasible combinations, given `end`(inclusive) and `include_night`"""

        m_asm: set[Solution] = set()
        for s1, s2 in itertools.product(solutions1, solutions2):
            s3 = s1 | s2
            if self.solution_possible(s3, end, include_night):
                m_asm.add(s3)

        return list(m_asm)

    def possible_matches_for_solution(
        self, sol: Solution, end: int, include_night: bool
    ) -> dict[str, list[str]]:
        """Given a solution, return a dict for the possible matches for the lefts that are not in the solution, given `end`(inclusive) and `includenight`"""

        sitting_lefts, sitting_rights, _ = get_candidates(sol)

        sitting_nomatches = self.season.get_sitting_no_matches(sol, end, include_night)

        mm = self.season.get_mm(end)
        pos_matches = {
            l: [
                r
                for r in self.season.rights
                if not self.no_match(l, r, end)
                and not (l, r) in sitting_nomatches
                and r != mm
                and r not in sitting_rights
            ]
            for l in self.season.lefts
            if l not in sitting_lefts
        }
        return pos_matches

    def merge_mm_in_solution(
        self, sol: Solution, partial_solutions: list[Solution]
    ) -> list[Solution]:
        """Merge solution with partial solutions to complete solution when the multiple match(es) are present the solution, given `end`(inclusive)"""

        _, _, mm_num = get_candidates(sol)
        if mm_num == 0:
            raise ValueError(
                "merge_mm_in_solution shouldn't be called if muliple match(es) are not in Solution"
            )

        sols = [sol.union(partial_solution) for partial_solution in partial_solutions]
        return sols

    def merge_mm_not_in_solution(
        self,
        sol: Solution,
        partial_solutions: list[Solution],
        end: int,
        include_night: bool,
    ) -> list[Solution]:
        """Merge solution with partial solutions to complete solutions when the multiple match(es) are not present the solution, given `end`(inclusive) and `includenight`"""

        mm = self.season.get_mm(end)

        sitting_nomatches = self.season.get_sitting_no_matches(sol, end, include_night)

        # compute possible partners left for missing rights
        addable_lefts = {
            r: [
                l
                for l in self.season.lefts
                if not (
                    self.no_match(l, r, end)
                    or (l, r) in sitting_nomatches
                    or (self.sm and l == self.sm)
                )
            ]
            for r in self.season.rights
        }

        solutions: list[Solution] = []

        for partial_solution in partial_solutions:
            base_matches = sol.union(partial_solution)

            _, sitting_rights = zip(*base_matches)
            # right that is missing in solution
            mr = next(r for r in self.season.rights if r not in sitting_rights)

            if mm:
                # mm is not None
                if mr == mm:
                    solutions.extend(
                        base_matches | {(l, mr)} for l in addable_lefts[mr]
                    )
                else:
                    dmleft = next(
                        l for (l, r) in base_matches if r == mm
                    )  # left that matches with mm
                    if dmleft not in addable_lefts[mr]:
                        continue
                    solutions.extend([base_matches | {(dmleft, mr)}])
            else:
                # Normalo 2023/24/25/26: dm not known, match mr with everyone possible
                solutions.extend([base_matches | {(l, mr)} for l in addable_lefts[mr]])

        return solutions

    def generate_complete_solutions(
        self, sol: Solution, end: int, include_night: bool
    ) -> list[Solution]:
        """Generate complete solutions by extending a partial solution.

        Args:
            sol (Solution): Partial solution to extend.
            end (int): Last episode to consider, inclusive.
            include_night (bool): Whether to consider the matching night of
                the last episode.

        Returns:
            list[Solution]: Complete solutions consistent with the given partial
                solution and the known constraints.
        """

        if not self.solution_possible(sol, end, include_night):
            return []
        if len(sol) == self.season.num_matches and self.solution_possible(
            sol, end, include_night
        ):
            return [sol]

        def zip_product(clefts: Iterable[str], ordering: Iterable[str]) -> Solution:
            return Solution(p for p in zip(clefts, ordering))

        _, _, mm_num = get_candidates(sol)
        pos_matches = self.possible_matches_for_solution(sol, end, include_night)

        # build pairs for lefts that are not in the given solution
        products = [
            list(ps)
            for ps in product_without_reps(
                list(pos_matches.values())
            )  # itertools.product(*pos_matches.values())
        ]
        partial_solutions = [zip_product(pos_matches.keys(), p) for p in products]

        if (self.season.has_two_dms and has_two_dms(sol)) or (
            not self.season.has_two_dms and mm_num == self.season.max_mm_size
        ):
            # Multiple match is already in Solution
            sols = self.merge_mm_in_solution(sol, partial_solutions)
            return sols

        solutions = self.merge_mm_not_in_solution(
            sol, partial_solutions, end, include_night
        )

        return solutions

    def generate_partial_solutions(
        self, end: int, include_night: bool
    ) -> list[Solution]:
        """Generate partial solutions from the matching night.

        For each matching night, the required number of matching pairs is selected and the resulting partial solutions are combined.
        Args:
            end (int): Last episode to consider, inclusive.
            include_night (bool): Whether to consider the matching night of
                the last episode.

        Returns:
            list[Solution]: Partial solutions consistent with the matching-night
                results and known constraints.
        """

        nights = self.season.get_nights(end)
        if not include_night:
            nights = nights[:-1]
        kpm = self.season.get_pms(end)

        pairs_per_night: list[list[Solution]] = []
        for night in nights:
            invalid_pairs = list(
                filter(lambda p: self.no_match(*p, end) or p in kpm, night.pairs)
            )
            definite_matches = list(filter(lambda p: p in kpm, night.pairs))
            remaining = set(night.pairs) - set(definite_matches) - set(invalid_pairs)

            combinations = [
                Solution(set(comb).union(kpm))
                for comb in itertools.combinations(
                    remaining, night.lights - len(definite_matches)
                )
            ]

            pairs_per_night.append(combinations)

        merged_solutions: list[Solution] = functools.reduce(
            lambda s1, s2: self.combine_possible_solutions(s1, s2, end, include_night),
            pairs_per_night,
        )

        return merged_solutions

    def solve(
        self, end: int, include_night: bool = True, validate: bool = False
    ) -> list[Solution]:
        """Solve season and return all possible solutions

        Args:
            end (int): Last episode to consider
            include_night (bool): Wheter the matching night of the last episode should be considered.Defaults to True.
            validate (bool, optional): Whether solutions should be vaidated by solution_possible after generating them. Defaults to False.

        Returns:
            list[Solution]: list of all possible solutions
        """

        # reduction of the search space by generating partial solutions from nights
        partial_solutions = self.generate_partial_solutions(end, include_night)

        solutions_unfiltered: set[Solution] = set()
        for g in partial_solutions:
            sols_g = self.generate_complete_solutions(g, end, include_night)
            solutions_unfiltered.update(sols_g)

        # the tests should guarantee that all generated solutions are possible
        if validate:
            solutions = [
                s
                for s in solutions_unfiltered
                if self.solution_possible(s, end, include_night)
            ]
        else:
            solutions = list(solutions_unfiltered)

        return list(solutions)
