from typing import override

from ayto.bitmask_solver import BitMaskSolver, partial_masks
from ayto.codec import RightBits, iter_bits

from .season import *
from .solver import TupleSolver, partial_sols


class DoubleMatchForRightSolver(TupleSolver):
    @override
    def find_multiple_match_violation(self, sol: PairSet, episode: int) -> str | None:

        supercheck = super().find_multiple_match_violation(sol, episode)
        if supercheck is not None:
            return supercheck

        pdict: dict[Person, list[Person]] = {r: [] for r in self.season.rights}
        for l, r in sol:
            pdict[r].append(l)

        mutiplers = [
            r for r in self.season.rights if len(pdict[r]) > 1
        ]  # rights with multiple matches
        if len(mutiplers) > 1:
            return "more_than_one_double_right_match"
        elif len(mutiplers) == 1:
            multir = mutiplers[0]
            multils = pdict[multir]
            if len(multils) == 2 and self.season.extra_left not in multils:
                return "extra_left_not_in_double_right_match"

        return None

    @override
    def generate_complete_solutions(
        self, sol: Solution, episode: int, include_night: bool
    ) -> list[Solution]:
        if not self.solution_possible(sol, episode, include_night):
            return []

        if len(sol) == self.season.num_matches and self.solution_possible(
            sol, episode, include_night
        ):
            return [sol]

        assert self.season.extra_left

        knowledge = self.knowledge_cache[(episode, include_night)]
        glefts, _, mmnum = get_candidates(sol)

        pos_matches: dict[Person, list[Person]] = self.possible_matches_for_solution(
            sol, episode, include_night
        )
        _ = pos_matches.pop(self.season.extra_left, None)
        if self.season.extra_left in glefts:
            # remove pair with self.sm if necessary
            el_matches = [r for l, r in sol if l == self.season.extra_left]
            # print(smmatches)
            # isol = set(sol)
            sol = Solution(
                sol.difference({(self.season.extra_left_name, el_matches[0])})
            )
        else:
            el_matches = [
                r
                for r in self.season.rights
                if not self.no_match(self.season.extra_left, r, episode, include_night)
            ]

        # products = [list(ps) for ps in product_without_reps(list(pos_matches.values()))]
        partial_solutions = partial_sols(pos_matches)
        if mmnum > 0:
            isols = self.merge_mm_in_solution(sol, partial_solutions)
        else:
            isols = self.merge_mm_not_in_solution(
                sol, partial_solutions, episode, include_night
            )

        # add self.sm as double_match
        sitting_no_matches = knowledge.get_sitting_no_matches(sol)
        solutions = [
            Solution(s | {(self.season.extra_left, r)})
            for r in el_matches
            for s in isols
            if (self.season.extra_left, r) not in sitting_no_matches
        ]

        return solutions


class DoubleMatchForRightBitMaskSolver(BitMaskSolver):
    @override
    def find_multiple_match_violation(self, sol: Mask, episode: int) -> str | None:
        codec = self.codec
        extra_left = self.season.extra_left
        assert extra_left

        supercheck = super().find_multiple_match_violation(sol, episode)
        if supercheck is not None:
            return supercheck

        multi_rs = codec.multi_rights(sol)  # rights with multiple matches
        n = multi_rs.bit_count()
        if n > 1:
            return "more_than_one_double_right_match"
        elif n == 1:
            # multir = multi_rs[0]
            mr = multi_rs.bit_length() - 1
            multils = codec.partners_of_right(sol, mr)
            if multils.bit_count() == 2 and not (codec.bit(extra_left, mr) & multils):
                return "extra_left_not_in_double_right_match"

        return None

    @override
    def generate_complete_solutions(
        self, sol: Mask, episode: int, include_night: bool
    ) -> list[Mask]:
        if not self.solution_possible(sol, episode, include_night):
            return []

        if sol.bit_count() == self.season.num_matches:
            return [sol]

        extra_left = self.season.extra_left
        assert extra_left
        codec = self.codec

        knowledge = self.knowledge_cache[(episode, include_night)]
        # glefts, _, mmnum = get_candidates(sol)

        pos_matches = self.possible_matches_for_solution(sol, episode, include_night)
        _ = pos_matches.pop(extra_left, None)

        el_match: RightBits = 0  #  all pairs of extra_left possible
        if codec.partners_of_left(sol, extra_left):
            # remove pair with self.sm if necessary
            el_match = codec.partners_of_left(sol, extra_left)

            # print(smmatches)
            # isol = set(sol)
            sol = sol & ~codec.row[extra_left]
        else:
            sitting_no_matches = knowledge.get_sitting_no_matches_mask(sol)
            el_match = codec.partners_of_left(
                codec.full_mask
                & ~self.no_match_masks[(episode, include_night)]
                & ~sitting_no_matches,
                extra_left,
            )

        # products = [list(ps) for ps in product_without_reps(list(pos_matches.values()))]
        partial_solutions = partial_masks(pos_matches, codec.R)
        mmnum = codec.max_mm_size(sol)
        if mmnum == 2:
            isols = self.merge_mm_in_solution(sol, partial_solutions)
        else:
            isols = self.merge_mm_not_in_solution(
                sol, partial_solutions, episode, include_night
            )

        # add self.sm as double_match
        # sitting_no_matches = knowledge.get_sitting_no_matches(sol)
        shift = extra_left * codec.R
        el_bits = [b << shift for b in iter_bits(el_match)]
        solutions = [s | p for p in el_bits for s in isols]

        return solutions
