import itertools
from collections.abc import Iterable
from typing import override

from ayto.base import Solver
from ayto.codec import Codec, Mask, RightBits, iter_bits
from ayto.season import BitMaskKnowledge, Pair, Season

Partial = tuple[Mask, RightBits]


def partial_masks(pos_matches: dict[int, RightBits], R: int) -> list[Partial]:
    # fewest options first, so dead ends disappear early
    items = sorted(pos_matches.items(), key=lambda kv: kv[1].bit_count())

    states: list[Partial] = [(0, 0)]  # (acc, used)
    for l, domain in items:
        shift = l * R
        states = [
            (acc | (rb << shift), used | rb)
            for acc, used in states
            for rb in iter_bits(domain & ~used)
        ]
    return states


class BitMaskSolver(Solver[Mask]):
    def __init__(self, season: Season) -> None:
        super().__init__(season)
        # self.season: Season = season
        self.codec: Codec = Codec(len(season.lefts), len(season.rights))

        self.knowledge_cache: dict[tuple[int, bool], BitMaskKnowledge] = (  # pyright: ignore[reportIncompatibleVariableOverride]
            {}
        )

        self.no_match_masks: dict[tuple[int, bool], Mask] = {}

        for i in range(self.season.max_lights + 1):
            for b in [True, False]:
                self.knowledge_cache[(i, b)] = BitMaskKnowledge(season, i, b)
                mask = 0
                for l, r in itertools.product(season.lefts, season.rights):
                    if self.no_match(l, r, i, b):
                        mask |= self.codec.bit(l, r)
                self.no_match_masks[(i, b)] = mask

    @override
    def from_pairs(self, pairs: Iterable[Pair]) -> Mask:
        return self.codec.encode(pairs)

    @override
    def no_match_reason(
        self, l: int, r: int, episode: int, include_night: bool
    ) -> str | None:
        knowledge = self.knowledge_cache[(episode, include_night)]
        codec = self.codec

        bit = codec.bit(l, r)
        kpm = knowledge.perfect_matches_mask
        pm_lefts = codec.lefts_in(kpm)
        pm_rights = codec.rights_in(kpm)
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
            codec.l_in_left_bits(pm_lefts, l) or codec.r_in_right_bits(pm_rights, r)
        ):
            # handle special case: for normalo2026, Alicia's match with Noel doesn't get revealed, despite Noel finding his match during the show
            if self.season.matchbox_reveals_dm:
                return "member_has_partner_already"
            else:
                # l cannot be part of multiple match
                if (
                    mm_person is not None
                    and not ((1 << r) & knowledge.mm_mask)  # ==
                    and self.no_match(l, mm_person, episode, include_night)
                ):
                    return "l_cannot_have_double_match"
                elif episode < self.season.mm_known_from_week:
                    return "cannot_be_in_double_match"

        return None

    @override
    def find_format_violation(self, sol: Mask) -> str | None:
        """Return the first format violation found in the solution, if any" """

        if sol.bit_count() > self.season.num_matches:
            return "too_many_matches"

        if not self.season.double_match_for_right and self.codec.multi_for_right(sol):
            return "double_match_for_right"

        return None

    @override
    def find_multiple_match_violation(self, sol: Mask, episode: int) -> str | None:
        """Return the first multiple-match violation found in the solution, if any"""
        codec = self.codec
        # lefts with multiple partners
        mult_lefts = codec.multi_lefts(sol)
        # double/triple match person
        mm_person = self.mm_person_cache[episode]

        if mult_lefts.bit_count() >= 2:
            return "too_many_double_matches"

        elif mult_lefts.bit_count() == 1:
            mult_left = codec.members(mult_lefts)[0]
            mult_rights = codec.partners_of_left(sol, mult_left)

            if mult_rights.bit_count() > self.season.max_mm_size:
                return "triple_match_not_allowed"

            if mm_person is not None:  # mm_person is not None
                seated_rights = codec.rights_in(sol)
                if not codec.r_in_right_bits(
                    codec.partners_of_left(sol, mult_left), mm_person
                ):
                    if mult_rights.bit_count() == self.season.max_mm_size:
                        return "mm_not_in_complete_multiple_match"
                    elif codec.r_in_right_bits(seated_rights, mm_person):
                        # only used in normalo2024
                        return "multiple_match_not_possible_without_mm"

        return None

    @override
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
        # if we don't have num_matches pairs, we allow lesser lights
        nights = knowledge.night_masks

        for night_mask, lights in nights:
            sol_lights: int = (sol & night_mask).bit_count()

            if sol_lights > lights:
                return "too_many_lights_in_night"
            elif sol_lights < lights and complete:
                return "not_enough_lights_in_night"

        return None

    @override
    def possible_matches_for_solution(
        self, sol: Mask, episode: int, include_night: bool
    ) -> dict[int, RightBits]:
        """Given a solution, return a dict for the possible matches for the lefts that are not in the solution, given `episode`(inclusive) and `include_night`"""
        codec = self.codec

        sitting_lefts = codec.lefts_in(sol)
        sitting_rights = codec.rights_in(sol)
        knowledge = self.knowledge_cache[(episode, include_night)]
        sitting_no_matches = knowledge.get_sitting_no_matches_mask(sol)

        not_sitting_lefts = ((1 << codec.L) - 1) & ~sitting_lefts  # difference
        pos_matches = {
            l: codec.full_row
            & (
                ~codec.partners_of_left(
                    self.no_match_masks[(episode, include_night)], l
                )
            )
            & (~codec.partners_of_left(sitting_no_matches, l))
            & (~knowledge.mm_mask)
            & (~sitting_rights)
            for l in codec.members(not_sitting_lefts)
        }
        return pos_matches

    @override
    def merge_mm_in_solution(
        self, sol: Mask, partial_solutions: list[Partial]
    ) -> list[int]:
        """Merge solution with partial solutions to complete solution when the multiple match(es) are present the solution, given `episode`(inclusive)"""
        mm_num = self.codec.max_mm_size(sol)
        if mm_num == 0:
            raise ValueError(
                "merge_mm_in_solution shouldn't be called if multiple match(es) are not in Solution"
            )

        sols = [sol | partial for partial, _ in partial_solutions]
        return sols

    @override
    def merge_mm_not_in_solution(
        self,
        sol: Mask,
        partial_solutions: list[Partial],
        episode: int,
        include_night: bool,
    ) -> list[Mask]:
        """Merge solution with partial solutions to complete solutions when the multiple match(es) are not present the solution, given `episode`(inclusive) and `include_night`"""
        codec = self.codec
        mm_person = self.mm_person_cache[episode]
        knowledge = self.knowledge_cache[(episode, include_night)]
        sitting_no_matches = knowledge.get_sitting_no_matches_mask(sol)

        # compute possible matches left for missing rights
        addable = (
            codec.full_mask
            & ~self.no_match_masks[(episode, include_night)]
            & ~sitting_no_matches
            & ~(codec.row[self.season.extra_left] if self.season.extra_left else 0)
        )
        addable_pairs = [
            list(iter_bits(codec.partners_of_right(addable, r)))
            for r in self.season.rights
        ]

        sitting_rights = codec.rights_in(sol)
        full = codec.full_row

        solutions: list[Mask] = []

        for partial, used in partial_solutions:
            base_matches = sol | partial
            free = full & ~(sitting_rights | used)

            # index of right that is missing in solution
            mr = free.bit_length() - 1
            if mm_person is not None:
                # mm_person is not None
                if mr == mm_person:
                    solutions.extend(base_matches | b for b in addable_pairs[mr])
                else:
                    dm_left = codec.left_of_pair_bit(
                        codec.partners_of_right(base_matches, mm_person)
                    )  # left that matches with mm
                    new = codec.bit(dm_left, mr)
                    if not addable & new:
                        continue
                    solutions.append(base_matches | codec.bit(dm_left, mr))
            else:
                # Normalo 2023/24/25/26: dm not known, match mr with everyone possible
                solutions.extend(base_matches | b for b in addable_pairs[mr])

        return solutions

    @override
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
            list[Mask]: Complete solutions consistent with the given partial
                solution and the known constraints.
        """

        if not self.solution_possible(sol, episode, include_night):
            return []
        if sol.bit_count() == self.season.num_matches:
            return [sol]
        codec = self.codec

        mm_num = codec.max_mm_size(sol)
        pos_matches = self.possible_matches_for_solution(sol, episode, include_night)
        partial_solutions = partial_masks(pos_matches, codec.R)

        if (
            self.season.two_dms
            and codec.multi_lefts(
                sol,
            ).bit_count()
            == 2
        ) or (not self.season.two_dms and mm_num == self.season.max_mm_size):
            # Multiple match is already in Solution
            sols = self.merge_mm_in_solution(sol, partial_solutions)
            return sols

        solutions = self.merge_mm_not_in_solution(
            sol, partial_solutions, episode, include_night
        )
        solutions = list(set(solutions))

        return solutions
