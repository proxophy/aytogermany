from typing import override

from ayto.bitmask_solver import BitMaskSolver, partial_masks
from ayto.codec import RightBits, iter_bits

from .season import *
from .tuple_solver import TupleSolver, partial_sols


class DoubleMatchForRightSolver(TupleSolver):

    def __init__(self, season: Season):
        super().__init__(season)
        if season.match_model != "double_match_for_right":
            raise ValueError(
                "DoubleMatchForRightSolver must be instantiated with season with match_model=double_match_for_right"
            )

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
            mmr = mutiplers[0]
            mmr_partners = pdict[mmr]
            if self.season.extra_left not in mmr_partners:
                return "extra_left_not_in_double_right_match"
           
            mml = mm_left(sol)
            if mml is not None:
                mml_partners = dict_rep(sol)[mml]
                # print(dict_rep(sol)[self.season.extra_left], mml, mml_partners)
                if any((self.season.extra_left, r) in sol for r in mml_partners):
                    return "double_matches_cannot_cross"

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

        extra_left = self.season.extra_left
        knowledge = self.knowledge_cache[(episode, include_night)]
        glefts, _, mmnum = get_candidates(sol)

        pos_matches: dict[Person, list[Person]] = self.possible_matches_for_solution(
            sol, episode, include_night
        )
        _ = pos_matches.pop(extra_left, None)
        if extra_left in glefts:
            # remove pair with self.sm if necessary
            el_matches = [r for l, r in sol if l == extra_left]
            sol = Solution(
                sol.difference({(extra_left, el_matches[0])})
            )
        else:
            el_matches = [
                r
                for r in self.season.rights
                if not self.no_match(extra_left, r, episode, include_night)
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
            Solution(s | {(extra_left, r)})
            for r in el_matches
            for s in isols
            if not ( (extra_left, r) in sitting_no_matches or r in mm_left_partners(s))
        ]

        return solutions


class DoubleMatchForRightBitMaskSolver(BitMaskSolver):
    def __init__(self, season: Season):
        super().__init__(season)
        if season.match_model != "double_match_for_right":
            raise ValueError(
                "DoubleMatchForRightBitMaskSolver must be instantiated with season with match_model=double_match_for_right"
            )

    @override
    def find_multiple_match_violation(self, sol: Mask, episode: int) -> str | None:
        codec = self.codec
        extra_left = self.season.extra_left

        supercheck = super().find_multiple_match_violation(sol, episode)
        if supercheck is not None:
            return supercheck

        multi_rs = codec.multi_rights(sol)  # rights with multiple matches
        n = multi_rs.bit_count()
        if n > 1:
            return "more_than_one_double_right_match"
        elif n == 1:
            # multir = multi_rs[0]
            mmr = multi_rs.bit_length() - 1
            mmr_partner = codec.partners_of_right(sol, mmr)
            if mmr_partner.bit_count() == 2 and not (codec.bit(extra_left, mmr) & mmr_partner):
                return "extra_left_not_in_double_right_match"

            mmls = codec.multi_lefts(sol)
            if mmls > 0:
                # we have a multiple left
                mml = mmls.bit_length() - 1
                mml_partners = codec.partners_of_left(sol, mml)
                el_partners = codec.partners_of_left(sol, extra_left)
                if mml_partners & el_partners:
                    return "double_matches_cannot_cross"


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
        
        solutions: list[Mask] = []
        for s in isols:
            ml = codec.multi_lefts(s).bit_length() - 1
            el_r = el_match & ~ codec.partners_of_left(s, ml)
            el_bits = [b << shift for b in iter_bits(el_r)]
            solutions.extend(s | p for p in el_bits)

        # solutions = [s | p for p in el_bits for s in isols]
        # solutions = [s for s in solutions if self.find_multiple_match_violation(s, episode)]

        return solutions
