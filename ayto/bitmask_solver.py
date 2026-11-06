import functools
import itertools
from collections.abc import Generator, Iterable, Sequence

from line_profiler import profile

from ayto.codec import Codec, Mask, RightBits
from ayto.season import BitMaskKnowledge, Season


def bitmask_product_without_reps(
    masks: Sequence[int],
) -> Generator[tuple[int, ...], None, None]:
    bits = [
        [1 << i for i in range(mask.bit_length()) if mask & (1 << i)] for mask in masks
    ]

    result = [0] * len(masks)

    def rec(i: int, used: int) -> Generator[tuple[int, ...], None, None]:
        if i == len(masks):
            yield tuple(result)
            return

        for bit in bits[i]:
            if used & bit:
                continue

            result[i] = bit
            yield from rec(i + 1, used | bit)

    yield from rec(0, 0)


class BitMaskSolver:

    def __init__(self, season: Season) -> None:
        self.season: Season = season
        self.codec: Codec = Codec(len(season.lefts), len(season.rights))

        self._no_match_cache: dict[tuple[int, int, int, bool], bool] = {}

        self.knowledge_cache: dict[tuple[int, bool], BitMaskKnowledge] = {}
        self.no_match_masks: dict[tuple[int, bool], Mask] = {}
        self.mm_person_cache: list[int | None] = [
            season.get_mm(ep) for ep in range(self.season.max_lights + 1)
        ]

        for i in range(self.season.max_lights + 1):
            for b in [True, False]:

                self.knowledge_cache[(i, b)] = BitMaskKnowledge(season, i, b)
                mask = 0
                for l, r in itertools.product(season.lefts, season.rights):
                    if self.no_match(l, r, i, b):
                        mask |= self.codec.bit(l, r)
                self.no_match_masks[(i, b)] = mask

    def no_match(self, l: int, r: int, end: int, include_night: bool) -> bool:
        """Return whether `(l, r)` is known not to be a match up to a given episode"""
        key = (l, r, end, include_night)
        cached = self._no_match_cache.get(key)
        if cached is None:
            cached = self.no_match_reason(l, r, end, include_night) is not None
            self._no_match_cache[key] = cached
        return cached

    def no_match_reason(
        self, l: int, r: int, episode: int, include_night: bool
    ) -> str | None:
        """Return the reason why `(l, r)` is known not to be a match up to a given episode"""
        knowledge = self.knowledge_cache[(episode, include_night)]

        bit = self.codec.bit(l, r)
        kpm = knowledge.perfect_matches_mask
        pm_lefts = self.codec.lefts_in(kpm)
        pm_rights = self.codec.rights_in(kpm)
        mm_person = self.mm_person_cache[episode]

        # matchbox result was false
        if bit & knowledge.mb_no_matches_mask:
            return "matchbox_false"

        # check for black out nights
        if bit & knowledge.bp_no_matches_mask:
            return "sitting_in_blackout_night"

        # one is part of known perfect match
        # consider multiple seated lefts
        # assumption: if dm is not known and you get double match,
        # you find the double match in the same episode
        if not (bit & knowledge.perfect_matches_mask) and (
            self.codec.l_in_left_bits(pm_lefts, l)
            or self.codec.r_in_right_bits(pm_rights, r)
        ):
            # handle special case: for normalo2026, Alicia's match with Noel doesnt get revealed, despite Noel finding his match during the show
            if self.season.matchbox_reveals_dm:
                return "member_has_partner_already"
            else:

                # l cannot be part of multiple match
                if (
                    mm_person
                    and not ((1 << r) & knowledge.mm_mask)  # ==
                    and self.no_match(l, mm_person, episode, include_night)
                ):
                    return "l_cannot_have_double_match"
                elif episode < self.season.mm_known_from_week:
                    return "cannot_be_in_double_match"

        return None

    def solution_possible(self, sol: Mask, episode: int, include_night: bool) -> bool:
        """Check whether the solution is feasible up to a given episode

        Args:
            sol: Solution containing the pairs to cneck
            episode: Last episode to consider
            include_night: Whether the matching night of the last episode should be considered

        Returns:
            `True` if no violation was found, ``False`` otherwise.
        """
        return self.find_violation(sol, episode, include_night) is None

    def find_format_violation(self, sol: Mask) -> str | None:
        """Return the first format violation found in the solution, if any" """

        if sol.bit_count() > self.season.num_matches:
            return "too_many_matches"

        if not self.season.double_match_for_right and self.codec.multi_for_right(sol):
            return "double_match_for_right"

        return None

    def find_multiple_match_violation(self, sol: Mask, episode: int) -> str | None:
        """Return the first multiple-match violation found in the solution, if any"""
        # sol_dict = {} #dict_rep(sol)
        # lefts with multiple partners
        mult_lefts = self.codec.multi_lefts(sol)
        # double/triple match person
        mm_person = self.mm_person_cache[episode]

        if mult_lefts.bit_count() >= 2:
            return "too_many_double_matches"

        elif mult_lefts.bit_count() == 1:
            mult_left = self.codec.members(mult_lefts)[0]
            mult_rights = self.codec.partners_of_left(sol, mult_left)

            if mult_rights.bit_count() > self.season.max_mm_size:
                return "triple_match_not_allowed"

            if mm_person is not None:  # mm_person is not None
                seated_rights = self.codec.rights_in(sol)
                if not self.codec.r_in_right_bits(
                    self.codec.partners_of_left(sol, mult_left), mm_person
                ):
                    if mult_rights.bit_count() == self.season.max_mm_size:
                        return "mm_not_in_complete_multiple_match"
                    elif self.codec.r_in_right_bits(seated_rights, mm_person):
                        # only used in normalo2024
                        return "multiple_match_not_possible_without_mm"

        return None

    def find_violation(
        self, sol: Mask, episode: int, include_night: bool
    ) -> str | None:
        """Return the first logic violation found in the solution, if any"""
        knowledge = self.knowledge_cache[(episode, include_night)]

        if (format_check := self.find_format_violation(sol)) is not None:
            return format_check

        # no known no matches
        if self.no_match_masks[(episode, include_night)] & sol:
            return "known_no_match"

        complete: bool = sol.bit_count() == self.season.num_matches

        # check conditions for double matches
        double_match_check = self.find_multiple_match_violation(sol, episode)
        if double_match_check is not None:
            return double_match_check

        # number of lights with pairs matching lights in nights
        # if we don't have self.nummatches pairs, we allow lesser lights
        nights = knowledge.night_masks

        for night_mask, lights in nights:
            sol_lights: int = (sol & night_mask).bit_count()

            if sol_lights > lights:
                return "too_many_lights_in_night"
            elif sol_lights < lights and complete:
                return "not_enough_lights_in_night"

        return None

    def combine_possible_solutions(
        self,
        solutions1: list[Mask],
        solutions2: list[Mask],
        episode: int,
        include_night: bool,
    ) -> list[int]:
        """Combines solutions from two lists and keeps feasible combinations, given `episode`(inclusive) and `include_night`"""

        m_asm: set[int] = set()
        for s1, s2 in itertools.product(solutions1, solutions2):
            s3 = s1 | s2
            if self.solution_possible(s3, episode, include_night):
                m_asm.add(s3)

        return list(m_asm)

    @profile
    def generate_partial_solutions(
        self, episode: int, include_night: bool
    ) -> list[Mask]:
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
        # todo (maybe): adapt this to bitmask? problem: vip2026 with missing lefts in sol

        knowledge = self.knowledge_cache[(episode, include_night)]
        nights = knowledge.nights
        kpm = knowledge.perfect_matches

        if len(nights) == 0:
            return [knowledge.perfect_matches_mask]

        pairs_per_night: list[list[int]] = []
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
                self.codec.encode(comb) | knowledge.perfect_matches_mask
                for comb in itertools.combinations(remaining, missinglights)
            ]

            pairs_per_night.append(combinations)

        merged_solutions: list[int] = functools.reduce(
            lambda s1, s2: self.combine_possible_solutions(
                s1, s2, episode, include_night
            ),
            pairs_per_night,
        )

        return merged_solutions

    def possible_matches_for_solution(
        self, sol: Mask, episode: int, include_night: bool
    ) -> dict[int, RightBits]:
        """Given a solution, return a dict for the possible matches for the lefts that are not in the solution, given `episode`(inclusive) and `includenight`"""

        sitting_lefts = self.codec.lefts_in(sol)
        sitting_rights = self.codec.rights_in(sol)
        knowledge = self.knowledge_cache[(episode, include_night)]
        sitting_nomatches = knowledge.get_sitting_no_matches_mask(sol)

        not_sitting_lefts = ((1 << self.codec.L) - 1) & ~sitting_lefts  # diffrence
        pos_matches = {
            l: self.codec.full_row
            & (
                ~self.codec.partners_of_left(
                    self.no_match_masks[(episode, include_night)], l
                )
            )
            & (~self.codec.partners_of_left(sitting_nomatches, l))
            & (~knowledge.mm_mask)
            & (~sitting_rights)
            for l in self.codec.members(not_sitting_lefts)
        }
        return pos_matches

    def merge_mm_in_solution(
        self, sol: Mask, partial_solutions: list[Mask]
    ) -> list[int]:
        """Merge solution with partial solutions to complete solution when the multiple match(es) are present the solution, given `episode`(inclusive)"""

        mm_num = self.codec.max_mm_size(sol)
        if mm_num == 0:
            raise ValueError(
                "merge_mm_in_solution shouldn't be called if muliple match(es) are not in Solution"
            )

        sols = [sol | partial_solution for partial_solution in partial_solutions]
        return sols

    def merge_mm_not_in_solution(
        self,
        sol: Mask,
        partial_solutions: list[Mask],
        episode: int,
        include_night: bool,
    ) -> list[Mask]:
        """Merge solution with partial solutions to complete solutions when the multiple match(es) are not present the solution, given `episode`(inclusive) and `includenight`"""

        mm_person = self.mm_person_cache[episode]
        knowledge = self.knowledge_cache[(episode, include_night)]
        sitting_nomatches = knowledge.get_sitting_no_matches_mask(sol)

        # compute possible partners left for missing rights
        addable_lefts_mask = (
            ((1 << self.codec.R * self.codec.L) - 1)
            & ~self.no_match_masks[(episode, include_night)]
            & ~sitting_nomatches
        )
        addable_lefts = {
            r: self.codec.members(
                self.codec.lefts_in(self.codec.partners_of_right(addable_lefts_mask, r))
            )
            for r in self.season.rights
        }
        # todo: add or (self.season.extra_left_name and l == self.season.extra_left)

        solutions: list[Mask] = []

        for partial_solution in partial_solutions:
            base_matches = sol | partial_solution

            # index of right that is missing in solution
            mr = self.codec.free_rights(base_matches).bit_length() - 1
            if mm_person is not None:
                # mm_person is not None
                if mr == mm_person:
                    solutions.extend(
                        base_matches | self.codec.bit(l, mr) for l in addable_lefts[mr]
                    )
                else:
                    dmleft = self.codec.left_of_pair_bit(
                        self.codec.partners_of_right(base_matches, mm_person)
                    )  # left that matches with mm
                    if dmleft not in addable_lefts[mr]:
                        continue
                    solutions.append(base_matches | self.codec.bit(dmleft, mr))
            else:
                # Normalo 2023/24/25/26: dm not known, match mr with everyone possible
                solutions.extend(
                    base_matches | self.codec.bit(l, mr) for l in addable_lefts[mr]
                )

        return solutions

    def generate_complete_solutions(
        self, sol: Mask, episode: int, include_night: bool
    ) -> list[Mask]:
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
        if sol.bit_count() == self.season.num_matches:
            return [sol]

        def zip_product(clefts: Iterable[int], ordering: Iterable[int]) -> int:
            mask = 0
            # check this
            for l, r in zip(clefts, ordering):
                mask |= r << (l * self.codec.R)
                # print(self.codec.pair_of_bit(r << (l * self.codec.R)) == (l, self.codec.members(r)[0]))
            return mask

        mm_num = self.codec.max_mm_size(sol)
        pos_matches = self.possible_matches_for_solution(sol, episode, include_night)
        products = [
            ps for ps in bitmask_product_without_reps(list(pos_matches.values()))
        ]

        partial_solutions = [zip_product(pos_matches.keys(), p) for p in products]

        if (
            self.season.two_dms
            and self.codec.multi_lefts(
                sol,
            )
            == 2
        ) or (not self.season.two_dms and mm_num == self.season.max_mm_size):
            # Multiple match is already in Solution
            sols = self.merge_mm_in_solution(sol, partial_solutions)
            return sols

        solutions = self.merge_mm_not_in_solution(
            sol, partial_solutions, episode, include_night
        )

        return solutions

    def solve(
        self, episode: int, include_night: bool = True, validate: bool = False
    ) -> list[Mask]:
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

        solutions_unfiltered: set[Mask] = set()
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
        return solutions
