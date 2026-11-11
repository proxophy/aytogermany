from collections import Counter
from typing import override

from ayto.bitmask_solver import BitMaskSolver, Partial
from ayto.codec import iter_bits

from .season import *
from .solver import TupleSolver


class TwoDoubleMatchesSolver(TupleSolver):
    @override
    def find_multiple_match_violation(self, sol: PairSet, episode: int) -> str | None:
        # VIP2025 allows two double matches, one of those must include mm_person

        seated_lefts = [l for l, _ in sol]
        seated_rights = [r for _, r in sol]
        sol_dict = dict_rep(sol)

        mult_lefts = [
            l for l, rs in sol_dict.items() if len(rs) > 1
        ]  # lefts with multiple partners
        mult_rights = [
            e for l in mult_lefts for e in sol_dict[l]
        ]  # rights part of multi match
        mm_person = self.mm_person_cache[episode]

        # if mm_person is seated and we have two double matches, he must be part of one of them
        if (
            mm_person in seated_rights
            and len(mult_lefts) == 2
            and mm_person not in mult_rights
        ):
            return "mm_person_not_in_double_match"

        r_counter = Counter(seated_lefts)
        # no triple matches
        if any(v > 2 for v in r_counter.values()):
            return "triple_match_not_allowed"
        return None

    @override
    def merge_mm_not_in_solution(
        self,
        sol: Solution,
        partial_solutions: list[tuple[Solution, set[Person]]],
        episode: int,
        include_night: bool,
    ) -> list[Solution]:
        knowledge = self.knowledge_cache[(episode, include_night)]
        sitting_no_matches = knowledge.get_sitting_no_matches(sol)
        # print("TwoDoubleMatchesSolver.merge_mm_not_in_solution", len(partial_solutions))
        addable_lefts = {
            r: [
                l
                for l in self.season.lefts
                if not (
                    self.no_match(l, r, episode, include_night)
                    or (l, r) in sitting_no_matches
                )
            ]
            for r in self.season.rights
        }

        solutions: list[Solution] = []

        mm_person = self.mm_person_cache[episode]
        _, sitting_rights, _ = get_candidates(sol)
        full = set(self.season.rights)

        for partial, used in partial_solutions:
            base = sol | partial
            free = list(full - used - sitting_rights)

            if len(free) == 2:
                # add two matches
                r1, r2 = free
                if r1 == mm_person or r2 == mm_person:
                    for l1 in addable_lefts[r1]:
                        for l2 in addable_lefts[r2]:
                            if l1 == l2:
                                continue

                            x = base | {(l1, r1), (l2, r2)}
                            solutions.append(x)
                else:
                    mm_person_partner = next(l for (l, r) in base if r == mm_person)
                    if mm_person_partner in addable_lefts[r1]:
                        solutions.extend(
                            base | {(mm_person_partner, r1), (l2, r2)}
                            for l2 in addable_lefts[r2]
                            if l2 != mm_person_partner
                        )
                    if mm_person_partner in addable_lefts[r2]:
                        solutions.extend(
                            base | {(mm_person_partner, r2), (l1, r1)}
                            for l1 in addable_lefts[r1]
                            if l1 != mm_person_partner
                        )
            elif len(free) == 1:
                mr = free[0]
                mml = mm_left(base)
                if mr == mm_person:
                    solutions.extend(
                        base | {(l, mr)} for l in addable_lefts[mr] if l != mml
                    )
                else:
                    mm_person_partner = next(l for (l, r) in base if r == mm_person)
                    d = dict_rep(base)
                    # dont add to mm_person if he's already part of double match
                    if len(d[mm_person_partner]) == 2:
                        solutions.extend(
                            base | {(l, mr)}
                            for l in addable_lefts[mr]
                            if l != mm_person_partner
                        )
                    else:
                        if mm_person_partner not in addable_lefts[mr]:
                            continue
                        solutions.append(base | {(mm_person_partner, mr)})
        return solutions


class TwoDoubleMatchesBitMaskSolver(BitMaskSolver):
    @override
    def find_multiple_match_violation(self, sol: Mask, episode: int) -> str | None:
        codec = self.codec
        R, full = codec.R, codec.full_row

        seated_rights: RightBits = 0
        mult_rights: RightBits = 0
        n_multi = 0
        has_triple = False
        for l in range(codec.L):
            row = (sol >> (l * R)) & full  # partners of left l
            seated_rights |= row
            if row.bit_count() > 1:
                n_multi += 1
                mult_rights |= row
                has_triple |= row.bit_count() > 2

        mm_person = self.knowledge_cache[(episode, True)].mm_mask
        if mm_person & seated_rights and n_multi == 2 and not (mm_person & mult_rights):
            return "mm_person_not_in_double_match"
        if has_triple:
            return "triple_match_not_allowed"
        return None

    @override
    def merge_mm_not_in_solution(
        self,
        sol: Mask,
        partial_solutions: list[Partial],
        episode: int,
        include_night: bool,
    ) -> list[Mask]:
        knowledge = self.knowledge_cache[(episode, include_night)]
        codec = self.codec
        sitting_no_matches = knowledge.get_sitting_no_matches_mask(sol)

        addable = (
            codec.full_mask
            & ~self.no_match_masks[(episode, include_night)]
            & ~sitting_no_matches
        )
        addable_pairs = [
            [
                (codec.left_of_pair_bit(p), p)
                for p in iter_bits(codec.partners_of_right(addable, r))
            ]
            for r in self.season.rights
        ]

        solutions: list[Mask] = []

        mm_bit = knowledge.mm_mask
        mm_person_idx = mm_bit.bit_length() - 1

        sitting_rights = codec.rights_in(sol)
        full = codec.full_row

        for partial, used in partial_solutions:
            base = sol | partial
            free: RightBits = full & ~(sitting_rights | used)
            n = free.bit_count()

            if n == 2:
                # index of free
                # r1, r2 = codec.members(free)
                low = free & -free  # returns lowest set bit as mask
                r1 = low.bit_length() - 1  # index of lowest bit
                r2 = (free ^ low).bit_length() - 1
                if free & mm_bit:
                    for l1, p1 in addable_pairs[r1]:
                        base1 = base | p1
                        solutions.extend(
                            base1 | p2 for l2, p2 in addable_pairs[r2] if l2 != l1
                        )
                else:
                    mm_person_pair = codec.partners_of_right(base, mm_person_idx)
                    mm_person_partner = codec.left_of_pair_bit(
                        mm_person_pair
                    )  # left with mm_person_
                    new1 = codec.bit(mm_person_partner, r1)
                    new2 = codec.bit(mm_person_partner, r2)
                    if addable & new1:
                        solutions.extend(
                            base | new1 | p2
                            for l2, p2 in addable_pairs[r2]
                            if l2 != mm_person_partner
                        )
                    if addable & new2:
                        solutions.extend(
                            base | new2 | p1
                            for l1, p1 in addable_pairs[r1]
                            if l1 != mm_person_partner
                        )
            elif n == 1:
                mr = free.bit_length() - 1  # free right
                mml = codec.multi_lefts(base)  # left that has double match
                if mr == mm_person_idx:
                    solutions.extend(
                        base | p
                        for l, p in addable_pairs[mr]
                        if not ((1 << l) & mml)
                        # base | {(l, mr)} for l in addable_pairs[mr] if l != mml
                    )
                else:
                    mm_person_pair = codec.partners_of_right(base, mm_person_idx)
                    mm_person_partner = codec.left_of_pair_bit(
                        mm_person_pair
                    )  # left with mm_person_
                    # dont add to mm_person if he's already part of double match
                    if codec.partners_of_left(base, mm_person_partner).bit_count() == 2:
                        solutions.extend(
                            base | p
                            for l, p in addable_pairs[mr]
                            if l != mm_person_partner
                        )
                    else:
                        new = codec.bit(mm_person_partner, mr)
                        if not (new & addable):
                            continue
                        solutions.append(base | new)
        return solutions
