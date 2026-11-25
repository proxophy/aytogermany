from typing import override

from ayto.base import Violation
from ayto.bitmask_solver import BitMaskSolver, partial_masks
from ayto.codec import Mask, RightBits, iter_bits
from ayto.season import (
    PairSet,
    Person,
    Season,
    Solution,
    dict_rep,
    get_candidates,
    has_dm_for_right,
    mm_left,
    mm_left_partners,
)
from ayto.tuple_solver import TupleSolver, partial_sols


class DoubleMatchForRightSolver(TupleSolver):

    def __init__(self, season: Season):
        super().__init__(season)
        if season.match_model != "double_match_for_right":
            raise ValueError(
                "DoubleMatchForRightSolver must be instantiated with season with match_model=double_match_for_right"
            )

    @override
    def find_multiple_match_violation(
        self, sol: PairSet, episode: int
    ) -> Violation | None:
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
            return Violation.MORE_THAN_ONE_DOUBLE_RIGHT_MATCH
        elif len(mutiplers) == 1:
            mmr = mutiplers[0]
            mmr_partners = pdict[mmr]
            n = len(mmr_partners)
            if n > 2:
                return Violation.RIGHT_MULTIPLE_MATCH_TOO_BIG

            if self.season.extra_left not in mmr_partners:
                return Violation.EXTRA_LEFT_NOT_IN_DOUBLE_RIGHT_MATCH

            mml = mm_left(sol)
            if mml is not None:
                mml_partners = dict_rep(sol)[mml]
                if any((self.season.extra_left, r) in sol for r in mml_partners):
                    return Violation.DOUBLE_MATCHES_CANNOT_CROSS

        return None

    @override
    def possible_matches_for_solution(
        self, sol: Solution, episode: int, include_night: bool
    ) -> dict[Person, list[Person]]:

        pos_matches = super().possible_matches_for_solution(sol, episode, include_night)
        sitting_lefts, _, _ = get_candidates(sol)
        knowledge = self.knowledge_cache[(episode, include_night)]
        sitting_no_matches = knowledge.get_sitting_no_matches(sol)
        mm_person = knowledge.mm

        extra_left = self.season.extra_left
        # print("possible_matches_for_solution")
        # print(extra_left in sitting_lefts, not has_dm_for_right(sol))

        # add right of extra_left back to candidates if we dont have a double match right
        if extra_left in sitting_lefts and not has_dm_for_right(sol):
            el_r: Person = next(r for l, r in sol if l == extra_left)
            for l in pos_matches:
                # print(
                #     l,
                #     el_r,
                #     (l, el_r) in sitting_no_matches,
                #     self.no_match(l, el_r, episode, include_night),
                #     el_r == mm_person,
                # )
                if (
                    (l, el_r) in sitting_no_matches
                    or self.no_match(l, el_r, episode, include_night)
                    or el_r == mm_person
                ):
                    continue

                pos_matches[l].append(el_r)

        return pos_matches

    @override
    def get_free(self, sol: Solution, used: set[Person])-> set[Person]:
        full = set(self.season.rights)
        sitting_lefts, sitting_rights, _ = get_candidates(sol)
        free = full - used - sitting_rights
        extra_left = self.season.extra_left
        if extra_left in sitting_lefts and not has_dm_for_right(sol):
            el_r: Person = next(r for l, r in sol if l == extra_left)
            free.add(el_r)
        return free

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
            el_matches = [r for l, r in sol if l == extra_left]
        else:
            el_matches = [
                r
                for r in self.season.rights
                if not self.no_match(extra_left, r, episode, include_night)
            ]

        partial_solutions = partial_sols(pos_matches)
        # print("DoubleMatchForRightSolver.generate_complete_solution")
        # print(len(partial_solutions))
        # for k, v in pos_matches.items():
        #     print(k, v)
        # print(sol, mmnum)

        if mmnum >= 2:
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
            if not ((extra_left, r) in sitting_no_matches or r in mm_left_partners(s))
        ]
        solutions = list(set(solutions))

        return solutions


class DoubleMatchForRightBitMaskSolver(BitMaskSolver):
    def __init__(self, season: Season):
        super().__init__(season)
        if season.match_model != "double_match_for_right":
            raise ValueError(
                "DoubleMatchForRightBitMaskSolver must be instantiated with season with match_model=double_match_for_right"
            )

    @override
    def find_multiple_match_violation(
        self, sol: Mask, episode: int
    ) -> Violation | None:
        codec = self.codec
        extra_left = self.season.extra_left

        supercheck = super().find_multiple_match_violation(sol, episode)
        if supercheck is not None:
            return supercheck

        multi_rs = codec.multi_rights(sol)  # rights with multiple matches
        n = multi_rs.bit_count()
        if n > 1:
            return Violation.MORE_THAN_ONE_DOUBLE_RIGHT_MATCH
        elif n == 1:
            # multir = multi_rs[0]
            mmr = multi_rs.bit_length() - 1
            mmr_partner = codec.partners_of_right(sol, mmr)
            n = mmr_partner.bit_count()
            if n > 2:
                return Violation.RIGHT_MULTIPLE_MATCH_TOO_BIG

            if not (codec.bit(extra_left, mmr) & mmr_partner):
                return Violation.EXTRA_LEFT_NOT_IN_DOUBLE_RIGHT_MATCH

            mmls = codec.multi_lefts(sol)
            if mmls > 0:
                # we have a multiple left
                mml = mmls.bit_length() - 1
                mml_partners = codec.partners_of_left(sol, mml)
                el_partners = codec.partners_of_left(sol, extra_left)
                if mml_partners & el_partners:
                    return Violation.DOUBLE_MATCHES_CANNOT_CROSS

        return None

    @override
    def possible_matches_for_solution(
        self, sol: Mask, episode: int, include_night: bool
    ) -> dict[Person, RightBits]:
        codec = self.codec
        pos_matches = super().possible_matches_for_solution(sol, episode, include_night)
        sitting_lefts = self.codec.lefts_in(sol)
        codec = self.codec
        knowledge = self.knowledge_cache[(episode, include_night)]
        sitting_no_matches = knowledge.get_sitting_no_matches_mask(sol)

        extra_left = self.season.extra_left
        # print("possible_matches_for_solution")
        # print(((1 << extra_left) & sitting_lefts), not codec.multi_for_right(sol))
        # add right of extra_left back to candidates if we dont have a double match right
        if ((1 << extra_left) & sitting_lefts) and not codec.multi_for_right(sol):
            el_r = codec.partners_of_left(sol, extra_left).bit_length() - 1

            for l in pos_matches:
                bit = codec.bit(l, el_r)
                # print(
                #     l,
                #     el_r,
                #     bit & sitting_no_matches,
                #     bit & self.no_match_masks[(episode, include_night)],
                #     (1 << el_r) & knowledge.mm_mask,
                # )
                if (
                    bit
                    & (
                        sitting_no_matches
                        | self.no_match_masks[(episode, include_night)]
                    )
                    or (1 << el_r) & knowledge.mm_mask
                ):
                    continue
                pos_matches[l] |=1 << el_r

        return pos_matches

    @override
    def get_free(self, sol: Mask, used: RightBits) -> RightBits:
        codec = self.codec
        sitting_rights = codec.rights_in(sol)
        
        free = super().get_free(sol, used)
        extra_left = self.season.extra_left
        if (1 << extra_left) & sitting_rights and not codec.multi_for_right(sol):
            el_r = codec.partners_of_left(sol, extra_left)
            free |= el_r
        return free
    

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

        pos_matches = self.possible_matches_for_solution(sol, episode, include_night)
        _ = pos_matches.pop(extra_left, None)

        el_matches: RightBits = 0  #  all pairs of extra_left possible
        if codec.partners_of_left(sol, extra_left):
            # remove pair with self.sm if necessary
            el_matches = codec.partners_of_left(sol, extra_left)
        else:
            sitting_no_matches = knowledge.get_sitting_no_matches_mask(sol)
            el_matches = codec.partners_of_left(
                codec.full_mask
                & ~self.no_match_masks[(episode, include_night)]
                & ~sitting_no_matches,
                extra_left,
            )

        partial_solutions = partial_masks(pos_matches, codec.R)
        # print("DoubleMatchForRightBitMaskSolver.generate_complete_solution")
        # print(len(partial_solutions))
        # for k, v in pos_matches.items():
        #     print(k, bin(v), codec.members(v))

        mmnum = codec.max_mm_size(sol)
        if mmnum >= 2:
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
            el_r = el_matches & ~codec.partners_of_left(s, ml)
            el_bits = [b << shift for b in iter_bits(el_r)]
            solutions.extend(s | p for p in el_bits)

        solutions = list(set(solutions))
        return solutions
