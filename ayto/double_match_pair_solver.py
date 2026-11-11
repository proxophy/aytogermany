from typing import override

from ayto.bitmask_solver import BitMaskSolver
from ayto.season import Pair

from .season import *
from .solver import TupleSolver


class DoubleMatchPairSolver(TupleSolver):
    @override
    def no_match_reason(
        self, l: Person, r: Person, episode: int, include_night: bool
    ) -> str | None:
        assert self.season.double_match_pair
        knowledge = self.knowledge_cache[(episode, include_night)]
        if episode >= self.season.double_match_pair_known:
            no_matches: list[Pair] = knowledge.mb_no_matches

            if r in self.season.double_match_pair:
                for dmr in self.season.double_match_pair:
                    if (l, dmr) in no_matches:
                        return "matchbox_false_for_double_match_pair"
        return super().no_match_reason(l, r, episode, include_night)

    @override
    def find_multiple_match_violation(self, sol: PairSet, episode: int) -> str | None:
        assert self.season.double_match_pair

        if episode >= self.season.double_match_pair_known:
            # lefts that partner with double_match_pair
            dml = [l for (l, r) in sol if r in self.season.double_match_pair]
            if len(dml) > 1 and dml[0] != dml[1]:
                return "double_match_pair_not_same_match"

            sol_dict = dict_rep(sol)

            # lefts with multiple partners
            mult_lefts = [l for l, rs in sol_dict.items() if len(rs) > 1]
            if len(mult_lefts) > 0:
                mult_left = mult_lefts[0]
                mult_rights = sol_dict[mult_left]
                if len(mult_rights) == 2 and set(self.season.double_match_pair) != set(
                    mult_rights
                ):
                    return "double_match_lefts_not_double_match_pair"

        return super().find_multiple_match_violation(sol, episode)

    @override
    def merge_mm_not_in_solution(
        self,
        sol: Solution,
        partial_solutions: list[tuple[Solution, set[Person]]],
        episode: int,
        include_night: bool,
    ) -> list[Solution]:
        assert self.season.double_match_pair

        if episode < self.season.double_match_pair_known:
            return super().merge_mm_not_in_solution(
                sol, partial_solutions, episode, include_night
            )

        solutions: list[Solution] = []
        knowledge = self.knowledge_cache[(episode, include_night)]
        sitting_no_matches = knowledge.get_sitting_no_matches(sol)
        full = set(self.season.rights)

        for partial, used in partial_solutions:
            ten_matches = sol.union(partial)
            _, sitting_rights = zip(*ten_matches)
            free = full - used - set(sitting_rights)

            # right that is missing in solution
            mr = free.pop()

            if mr not in self.season.double_match_pair:
                continue
            # find left that pairs with one person of double_match_pair
            dm_left = next(
                l for (l, r) in ten_matches if r in self.season.double_match_pair
            )
            if (dm_left, mr) in sitting_no_matches:
                continue
            solutions.append(ten_matches | {(dm_left, mr)})

        return solutions


class DoubleMatchPairBitMaskSolver(BitMaskSolver):
    @override
    def no_match_reason(
        self, l: Person, r: Person, episode: int, include_night: bool
    ) -> str | None:
        assert self.season.double_match_pair
        codec = self.codec
        knowledge = self.knowledge_cache[(episode, include_night)]
        if episode >= self.season.double_match_pair_known:
            if knowledge.mb_no_matches_mask & codec.bit(l, r):
                return "matchbox_false"

            mask = codec.encode({(l, dmr) for dmr in self.season.double_match_pair})
            if (
                r in self.season.double_match_pair
                and mask & knowledge.mb_no_matches_mask
            ):
                return "matchbox_false_for_double_match_pair"

        return super().no_match_reason(l, r, episode, include_night)

    @override
    def find_multiple_match_violation(self, sol: Mask, episode: int) -> str | None:
        assert self.season.double_match_pair
        codec = self.codec

        if episode >= self.season.double_match_pair_known:
            dmr1, dmr2 = self.season.double_match_pair
            # lefts that partner with double_match_pair
            # dml = [l for (l, r) in sol if r in self.season.double_match_pair]
            dml = codec.partners_of_right(sol, dmr1) | codec.partners_of_right(
                sol, dmr2
            )
            if dml.bit_count() > 1 and codec.lefts_in(m=dml).bit_count() > 1:
                return "double_match_pair_not_same_match"

            # lefts with multiple partners
            mult_lefts = codec.multi_lefts(
                sol
            )  # [l for l, rs in sol_dict.items() if len(rs) > 1]
            if mult_lefts.bit_count() > 0:
                mult_left = mult_lefts.bit_length() - 1  # first multi left
                mult_rights = codec.partners_of_left(
                    sol, mult_left
                )  # rights in multi match
                if mult_rights.bit_count() == 2 and mult_rights != (
                    (1 << dmr1) | (1 << dmr2)
                ):
                    return "double_match_lefts_not_double_match_pair"

        return super().find_multiple_match_violation(sol, episode)

    @override
    def merge_mm_not_in_solution(
        self,
        sol: Mask,
        partial_solutions: list[tuple[Mask, RightBits]],
        episode: int,
        include_night: bool,
    ) -> list[Mask]:
        assert self.season.double_match_pair
        codec = self.codec

        if episode < self.season.double_match_pair_known:
            return super().merge_mm_not_in_solution(
                sol, partial_solutions, episode, include_night
            )

        solutions: list[Mask] = []
        knowledge = self.knowledge_cache[(episode, include_night)]
        sitting_no_matches = knowledge.get_sitting_no_matches_mask(sol)
        full = codec.full_row
        sitting_rights = codec.rights_in(sol)
        dmr1, dmr2 = self.season.double_match_pair

        for partial, used in partial_solutions:
            base_matches = sol | partial
            free = full & ~(sitting_rights | used)

            # right that is missing in solution
            mr = free.bit_length() - 1

            if mr not in self.season.double_match_pair:
                continue

            # find pair of other in double match pair
            dm_pair = codec.partners_of_right(
                base_matches, dmr1
            ) | codec.partners_of_right(base_matches, dmr2)
            dm_left = codec.left_of_pair_bit(dm_pair)
            new = codec.bit(dm_left, mr)
            if new & sitting_no_matches:
                continue
            solutions.append(base_matches | new)

        return solutions
