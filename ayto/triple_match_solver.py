from collections import Counter
from typing import override

from ayto.bitmask_solver import BitMaskSolver
from ayto.codec import RightBits

from .season import *
from .tuple_solver import TupleSolver


class TripleMatchSolver(TupleSolver):
    def __init__(self, season: Season):
        super().__init__(season)
        if season.match_model != "triple_match":
            raise ValueError(
                "TripleMatchSolver must be instantiated with season with match_model=triple_match"
            )

    @override
    def no_match_reason(
        self, l: Person, r: Person, episode: int, include_night: bool
    ) -> str | None:
        """Return whether a pair is known not to be a match up to `end`"""
        knowledge = self.knowledge_cache[(episode, include_night)]
        kpm = knowledge.perfect_matches
        pm_lefts = [a for a, _ in kpm]
        pm_rights = {b for _, b in kpm}

        counts = Counter(pm_lefts)
        if (l, r) not in kpm and l in pm_lefts:
            # lefts with multiple matches from matchboxes, appear x times
            # for x perfect matches
            mmls = [p for p in pm_lefts if counts[p] > 1]
            # Normalo 24
            if len(mmls) < 3 and l in mmls and r not in pm_rights:
                # we only know two out of three of the multiple matches in Normalo 2024 at episode 10
                return None
            return "triple_match_already_exists"

        return super().no_match_reason(l, r, episode, include_night)

    @override
    def merge_mm_not_in_solution(
        self,
        sol: Solution,
        partial_solutions: list[tuple[Solution, set[Person]]],
        episode: int,
        include_night: bool,
    ) -> list[Solution]:
        # for documentation refer to Solver.merge_mm_not_in_solution
        knowledge = self.knowledge_cache[(episode, include_night)]
        sitting_no_matches = knowledge.get_sitting_no_matches(sol)

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
        mm_person = knowledge.mm

        for partial, _ in partial_solutions:
            base_matches = sol.union(partial)
            sitting_lefts, sitting_rights = zip(*base_matches)
            mrs = [r for r in self.season.rights if r not in sitting_rights]

            if len(mrs) == 2:
                mr1, mr2 = mrs
                if mm_person in mrs:
                    # no multiple seatings yet , Mela not seated
                    for l in self.season.lefts:
                        if l in addable_lefts[mr1] and l in addable_lefts[mr2]:
                            solutions.append(base_matches | {(l, mr1), (l, mr2)})
                else:
                    # no mutiple seatings, Mela seated
                    dmleft = next(
                        l for (l, r) in base_matches if r == mm_person
                    )  # find partner of Mela
                    if dmleft in addable_lefts[mr1] and dmleft in addable_lefts[mr2]:
                        solutions.append(base_matches | {(dmleft, mr1), (dmleft, mr2)})

            elif len(mrs) == 1:
                counter = Counter(sitting_lefts)
                # left that already has two matches
                smleft = next(l for l in self.season.lefts if counter[l] == 2)
                # missing right
                mr = next(r for r in self.season.rights if r not in sitting_rights)
                if (
                    self.no_match(smleft, mr, episode, include_night)
                    or (smleft, mr) in sitting_no_matches
                ):
                    continue
                solutions.append(base_matches | {(smleft, mr)})

        return solutions


class TripleMatchBitMaskSolver(BitMaskSolver):

    def __init__(self, season: Season):
        super().__init__(season)
        if season.match_model != "triple_match":
            raise ValueError(
                "TripleMatchBitMaskSolver must be instantiated with season with match_model=triple_match"
            )

    @override
    def no_match_reason(
        self, l: Person, r: Person, episode: int, include_night: bool
    ) -> str | None:
        """Return whether a pair is known not to be a match up to `end`"""
        knowledge = self.knowledge_cache[(episode, include_night)]
        codec = self.codec

        kpm = knowledge.perfect_matches_mask
        pm_lefts = codec.lefts_in(kpm)
        pm_rights = codec.rights_in(kpm)

        if not (codec.bit(l, r) & kpm) and (1 << l & pm_lefts):
            # lefts with multiple matches from matchboxes, appear x times
            # for x perfect matches
            mmls = codec.multi_lefts(kpm)
            # Normalo 24
            if (
                codec.max_mm_size(kpm) < 3
                and ((1 << l) & mmls)
                and not ((1 << r) & pm_rights)
            ):
                # we only know two out of three of the multiple matches in Normalo 2024 at end of episode 4
                return None
            return "triple_match_already_exists"

        return super().no_match_reason(l, r, episode, include_night)

    @override
    def merge_mm_not_in_solution(
        self,
        sol: Mask,
        partial_solutions: list[tuple[Mask, RightBits]],
        episode: int,
        include_night: bool,
    ) -> list[Mask]:
        knowledge = self.knowledge_cache[(episode, include_night)]
        codec = self.codec

        sitting_no_matches = knowledge.get_sitting_no_matches_mask(sol)

        # compute possible matches left for missing rights
        addable = (
            codec.full_mask
            & ~self.no_match_masks[(episode, include_night)]
            & ~sitting_no_matches
        )

        solutions: list[Mask] = []
        mm_bit: RightBits = knowledge.mm_mask
        mm_idx = mm_bit.bit_length() - 1
        sitting_rights = codec.rights_in(sol)
        full = codec.full_row
        first_col = codec.first_col

        for partial, used in partial_solutions:
            base = sol | partial
            free: RightBits = full & ~(sitting_rights | used)
            n = free.bit_count()

            if n == 2:
                low = free & -free  # returns lowest set bit as mask
                r1 = low.bit_length() - 1  # index of lowest bit
                r2 = (
                    free ^ low
                ).bit_length() - 1  # index of highest (other) bit in free

                # bit l*R is set iff (l, r1) and (l, r2) are both addable
                # addable >> r1 puts pairs at (l1, r1) to (l1, 0)
                rows = (addable >> r1) & (addable >> r2) & first_col

                if not (free & mm_bit):
                    # mm already seated
                    partner = codec.left_of_pair_bit(
                        codec.partners_of_right(base, mm_idx)
                    )
                    rows &= 1 << (partner * codec.R)

                while rows:
                    b: Mask = rows & -rows  # lowest bit
                    rows ^= b
                    solutions.append(base | (b << r1) | (b << r2))

            elif n == 1:
                mr = free.bit_length() - 1  # free right
                mml = (
                    codec.multi_lefts(base).bit_length() - 1
                )  # left that has double match

                p1 = codec.bit(mml, mr)
                if p1 & addable:
                    solutions.append(base | p1)

        return solutions
