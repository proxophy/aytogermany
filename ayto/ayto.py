import itertools
import functools

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


def product_without_reps(arr: list[list]):
    used = set()
    current = []

    def rec(i):
        if i == len(arr):
            yield current.copy()
            return

        for x in arr[i]:
            if x in used:
                continue

            used.add(x)
            current.append(x)

            yield from rec(i + 1)

            current.pop()
            used.remove(x)

    yield from rec(0)


class Solver:
    """Solver for a typical AYTO season with one double match.
    Season-specific solvers inhert from this solver and can override its logic where necessary.
    Args:
        season (Season): Season object containing candidates, match boxes, matching nights and other information about the season
    """

    season: Season
    sm = None
    double_match_for_right: bool = False
    two_dms: bool = False

    def __init__(self, season: Season) -> None:
        self.season: Season = season
        self.bo_nights = self.season.get_black_out_nights()

    @functools.cache
    def no_match(self, l: str, r: str, end: int) -> bool:
        """Return whether a pair is known not to be a match up to `end`"""
        end = max(0, min(end, 10))

        mb = self.season.get_matchboxes(end)
        kpm = self.season.get_pms(end)
        hpm = [e for pp in kpm for e in pp]

        # matchbox result was false
        if (l, r) in mb and not mb[(l, r)]:
            return True

        # check for black out nights
        for bo in self.bo_nights:
            if bo > end:
                break
            if (l, r) in self.season.nights[bo].pairs and (l, r) not in kpm:
                return True

        # one is part of known perfect match
        # consider multiple seated lefts
        # assumption: if dm is not known and you get double match,
        # you find the double match in the same episode
        if (l, r) not in kpm and (l in hpm or r in hpm):
            # handle special case: for normalo2026, Alicia's match with Noel doesnt get revealed, despite Noel finding his match during the show
            if self.season.mb_reveals_dm or r != self.season.get_mm():
                return True

        return False

    def solution_has_correct_format(self, sol: Solution | set[Pair]) -> dict:
        """Check whether a solution has the expected format.
        Returns dict with keys `res`, `reason` and optionally `detail`
        """

        if len(sol) > self.season.num_matches:
            return {"res": False, "reason": "too_many_matches"}

        if not self.double_match_for_right and double_match_for_right(sol):
            return {"res": False, "reason": "double_match_for_right"}

        for l, r in sol:
            if l not in self.season.lefts:
                return {
                    "res": False,
                    "reason": "wrongly_written_names_as_left",
                    "detail": l,
                }
            elif r not in self.season.rights:
                return {
                    "res": False,
                    "reason": "wrongly_written_names_as_right",
                    "detail": r,
                }

        return {"res": True, "reason": ""}

    def solution_possible(
        self, sol: Solution | set[Pair], end: int, include_night: bool
    ) -> dict:
        """Check whether the solution is feasible up to a given episode

        Args:
            sol (Solution): Solution containing the pairs to cneck
            end (int): Last episode to consider
            include_night (bool): Wheter the matching night of the last episode should be considered

        Returns:
            dict: Result of the feasibility check, containing:
            - ``res`` (bool): Whether the solution is feasible.
            - ``reason`` (str): Reason for the result.
            - ``detail``: Additional details about the result, such as
              the pair that caused the solution to be rejected.
        """

        if len(sol) > self.season.num_matches:
            return {"res": False, "reason": "too_many_matches"}

        format_check = self.solution_has_correct_format(sol)
        if not format_check["res"]:
            return format_check

        end = max(0, min(end, 10))

        # no known no matches
        for pair in sol:
            if self.no_match(*pair, end):
                return {
                    "res": False,
                    "reason": "known_no_match",
                    "detail": pair,
                }

        complete: bool = len(sol) == self.season.num_matches

        # check conditions for double matches
        double_match_logic = self.check_multiple_match_logic(sol, end)
        if not double_match_logic["res"]:
            return double_match_logic

        # number of lights with pairs matching lights in nights
        # if we don't have self.nummatches pairs, we allow lesser lights
        nights = self.season.get_nights(end)
        if not include_night:
            nights = nights[:-1]
        i = 0
        for night in nights:
            sol_lights: int = len(sol & night.pairs)

            if sol_lights > night.lights:
                return {"res": False, "reason": "too_many_lights_in_night", "detail": i}
            elif sol_lights < night.lights:
                if complete:
                    return {
                        "res": False,
                        "reason": "not_enough_lights_in_night",
                        "detail": i,
                    }
            i += 1

        return {"res": True, "reason": ""}

    def check_multiple_match_logic(self, sol: Solution | set[Pair], end: int) -> dict:
        """Check whether a solution satisfies the multiple-match constraints up to a given episode.
        Returns dict with keys `res`, `reason` and optionally `detail`
        """

        sol_dict = dict_rep(sol)
        # lefts with multiple partners
        mult_lefts = [l for l, rs in sol_dict.items() if len(rs) > 1]
        # double/tripple match person
        mm = self.season.get_mm(end)

        if len(mult_lefts) >= 2:
            return {"res": False, "reason": "too_many_double_matches"}

        elif len(mult_lefts) == 1:
            mult_left = mult_lefts[0]
            mult_rights = sol_dict[mult_left]
            if len(mult_rights) > self.season.max_mm_size:
                return {
                    "res": False,
                    "reason": "triple_match_not_allowed",
                }

            if mm:  # mm is not None
                _, seated_rights = zip(*sol)

                if (
                    len(mult_rights) == self.season.max_mm_size
                    and mm not in sol_dict[mult_left]
                ):
                    return {
                        "res": False,
                        "reason": "mm_not_in_complete_multiple_match",
                        "detail": (mult_left, mult_rights),
                    }
                elif mm in seated_rights and mm not in sol_dict[mult_left]:
                    # only used in normalo2024
                    return {
                        "res": False,
                        "reason": "multiple_match_not_possible_without_mm",
                        "detail": (mult_left, mult_rights),
                    }

        return {"res": True, "reason": ""}

    def combine_possible_solutions(
        self,
        solutions1: list[Solution],
        solutions2: list[Solution],
        end: int,
        include_night: bool,
    ) -> list[Solution]:
        """Combines solutions from two lists and keeps feasible combinations, given `end`(inclusive) and `include_night`"""

        m_asm = set()
        for s1, s2 in itertools.product(solutions1, solutions2):
            s3 = s1 | s2
            pred = self.solution_possible(s3, end, include_night)
            if pred["res"]:
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

        solutions = []

        for partial_solution in partial_solutions:
            base_matches = sol.union(partial_solution)

            _, sitting_rights = zip(*base_matches)
            # right that is missing in solution
            mr = [r for r in self.season.rights if r not in sitting_rights][0]

            if mm:
                # mm is not None
                if mr == mm:
                    solutions.extend(
                        base_matches | {(l, mr)} for l in addable_lefts[mr]
                    )
                else:
                    dmleft = [l for (l, r) in base_matches if r == mm][
                        0
                    ]  # left that matches with mm
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

        if not self.solution_possible(sol, end, include_night)["res"]:
            return []
        if (
            len(sol) == self.season.num_matches
            and self.solution_possible(sol, end, include_night)["res"]
        ):
            return [sol]

        def zip_product(clefts, ordering) -> Solution:
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
        partial_solutions = list(
            map(lambda p: zip_product(pos_matches.keys(), p), products)
        )
        # print(mm_num,self.season.max_mm_size, dict_rep(sol))
        if (self.two_dms and has_two_dms(sol)) or (not self.two_dms and mm_num == self.season.max_mm_size):
            # Multiple match is already in Solution
            sols = self.merge_mm_in_solution(sol, partial_solutions)
            print("merge_in", has_two_dms(sol), self.two_dms)
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

        pairs_per_night = []
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
            solutions = list(
                s
                for s in solutions_unfiltered
                if self.solution_possible(s, end, include_night)["res"]
            )
        else:
            solutions = list(solutions_unfiltered)

        return list(solutions)
