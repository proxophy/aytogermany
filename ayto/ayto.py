from collections import Counter
import itertools
import functools

from .models import *
from .utils import product_without_reps, listeq


class Solver:
    season: Season

    def __init__(self, season: Season) -> None:
        self.season: Season = season

    def no_match(self, p: Pair, end: int) -> bool:
        """(l,r) are definitely no match"""
        assert isinstance(p, Pair)
        assert (
            p.l in self.season.lefts and p.r in self.season.rights
        ), f"l: {p.l}, r: {p.r}"
        end = max(0, min(end, 10))

        mb = self.season.get_matchboxes(end)
        kpm = self.season.get_pms(end)
        hpm = [e for pp in kpm for e in pp]

        # matchbox result was false
        if p in mb and not mb[p]:
            return True

        # one is part of known perfect match
        # consider multiple seated lefts
        # assumption: if dm is not known and you get double match,
        # you find the double match in the same episode
        if p not in kpm and (p.l in hpm or p.r in hpm) and p.r != self.season.mm:
            return True

        return False

    def solution_correct_format(self, sol: Solution) -> bool:
        assert isinstance(sol, Solution)

        if len(sol) > self.season.nummatches:
            print("len(solution) > self.nummatches")
            return False

        ls, rs = zip(*sol)
        if 2 in Counter(rs).values():
            # print(f"No double matches for a right")
            return False
        if any(l not in self.season.lefts for l in ls):
            print(
                f"Wrongly written names: {[c for c in ls if c not in self.season.lefts]}"
            )
            return False
        if any(r not in self.season.rights for r in rs):
            print(
                f"Wrongly written names: {[c for c in rs if c not in self.season.rights]}"
            )
            return False

        return True

    def solution_possible(self, sol: Solution, end: int) -> bool:
        assert isinstance(sol, Solution)
        if len(sol) > self.season.nummatches:
            return False
        end = max(0, min(end, 10))

        if not self.solution_correct_format(sol):
            return False


        complete: bool = len(sol) == self.season.nummatches

        # no known no matches
        if any([self.no_match(p, end) for p in sol]):
            ba: list = [self.no_match(p, end) for p in sol]
            trueindex = ba.index(True)
            print(f"Solution has known no match: {list(sol)[trueindex]}")
            return False

        # VIP 23
        # perfect matches of dmtuple must be the same person
        if self.season.dmtuple is not None and end >= self.season.dmtupleknown:
            dml = [l for (l, r) in sol if r in self.season.dmtuple]
            if len(dml) > 1 and dml[0] != dml[1]:
                # print(f"dmtuple rights do not have same pm {len(sol)} {dml}")
                return False

        # check condition for double matches
        pdict = {l: [] for l in self.season.lefts}
        for l, r in sol:
            pdict[l].append(r)
        g_lefts, g_rights = zip(*sol)

        # lefts with multiple matches
        mutiplels = [l for l in self.season.lefts if len(pdict[l]) > 1]

        if len(mutiplels) == 2:
            if not self.season.two_dms:
                # print("Only one double/tripple match")
                return False
            else:
                # VIP 2025: two double matches
                multiplers = [e for l in mutiplels for e in pdict[l]]
                if not self.season.mm in multiplers:
                    return False
        elif len(mutiplels) == 1:
            mutiplel = mutiplels[0]
            multiplers = pdict[mutiplel]

            if (
                (self.season.dmtuple is not None and end >= self.season.dmtupleknown)
                and len(multiplers) == 2
                and set(self.season.dmtuple) != set(multiplers)
            ):
                return False
            elif (
                self.season.mm is not None
                and len(multiplers) == 2
                and self.season.mm not in pdict[mutiplel]
                and not self.season.two_dms
            ):
                return False
        elif len(mutiplels) > 2:
            return False

        # if Solution has nummatches matches
        if complete:
            # we must have 10 seated lefts and nummatches rights
            g_lefts, g_rights = zip(*sol)
            if len(set(g_lefts)) != len(self.season.lefts) or len(set(g_rights)) != len(
                self.season.rights
            ):
                return False

        # number of lights with pairs matching lights in nights
        # if we don't have self.nummatches pairs, we allow lesser lights
        nights = self.season.get_nights(end)
        i = 0
        for night in nights:
            clights = sol.intersectionlength(night.pairs)
            # print(f"checking night {i} with {lights} lights, Solution has {clights} lights")
            # print(intersection)
            # print(len(pairs), pairs)
            i += 1
            if clights > night.lights:
                # print(clights, ">", night.lights)
                return False
            elif clights < night.lights:
                # print(clights, "<", night.lights)
                if complete:
                    return False

        return True

    def merge_solutions_lists(
        self, psl_1: list[Solution], psl_2: list[Solution], end: int = 10
    ):
        """
        Output: list of merged together partial sols
        """
        m_asm = []
        for g1, g2 in itertools.product(psl_1, psl_2):
            pred = self.solution_possible(g1.union(g2), end)
            if pred and g1.union(g2) not in m_asm:
                m_asm.append(g1.union(g2))

        return m_asm

    def get_solution_leftrights(self, sol: Solution):
        assert isinstance(sol, Solution)
        if len(sol) > 0:
            g_lefts, g_rights = zip(*sol)
            g_lefts, g_rights = list(g_lefts), list(g_rights)
        else:
            g_lefts, g_rights = set(), set()
        return set(g_lefts), set(g_rights), len(g_lefts) - len(set(g_lefts)) > 0

    def possible_matches_for_solution(
        self, sol: Solution, end: int
    ) -> dict[str, list[str]]:
        assert isinstance(sol, Solution)
        g_lefts, g_rights, _ = self.get_solution_leftrights(sol)

        nights = self.season.get_nights(end)
        sitting_nomatches = {}
        # Consider sitting matches as no matches if not in partial sol todo: remove duplicate
        for night in nights:
            for p in set(night.pairs) - set(sol.pairs):
                sitting_nomatches[p] = True

        pos_matches = {
            l: [
                r
                for r in set(self.season.rights) - g_rights
                if not self.no_match(Pair(l, r), end)
                and not sitting_nomatches.get(Pair(l, r), False)
                and r != self.season.mm
            ]
            for l in self.season.lefts
            if l not in g_lefts
        }
        return pos_matches

    def merge_mm_in_solution(
        self, sol: Solution, other_matches_list: list[Solution], end: int
    ) -> list[Solution]:
        assert isinstance(sol, Solution)
        _, _, dm_in_psol = self.get_solution_leftrights(sol)
        assert (
            dm_in_psol
        ), "merge_mm_in_Solution shouldn't be called if muliple match(es) are not in Solution"
        sols = [sol.union(othermatches) for othermatches in other_matches_list]
        return sols

    def merge_mm_not_in_solution(
        self, sol: Solution, other_matches_list: list[Solution], end: int
    ) -> list[Solution]:
        assert isinstance(sol, Solution)

        nights = self.season.get_nights(end)
        sitting_nomatches: dict[Pair, bool] = {}
        # Consider sitting matches as no matches if not in Solution and we have the right
        # amount of lights
        for night in nights:
            pl = sol.intersectionlength(night.pairs)
            if pl > night.lights:
                return []
            elif pl < night.lights:
                continue
            for p in set(night.pairs) - set(sol.pairs):
                sitting_nomatches[p] = True

        solutions = []
        addmatches_dict = {
            r: [
                Pair(l, r)
                for l in self.season.lefts
                if not (
                    self.no_match(Pair(l, r), end)
                    or sitting_nomatches.get(Pair(l, r), False)
                )
            ]
            for r in self.season.rights
        }

        for othermatches in other_matches_list:
            tenmatches = sol.union(othermatches)
            assert len(tenmatches) == 10

            _, crights = zip(*tenmatches)
            mr = [r for r in self.season.rights if r not in crights][0]  # missing right

            # VIP 2023: dmtuple
            if self.season.dmtuple is not None and end >= self.season.dmtupleknown:
                if mr not in self.season.dmtuple:
                    continue

                dmleft = [l for (l, r) in tenmatches if r in self.season.dmtuple][0]
                solutions.append(tenmatches.addpair(Pair(dmleft, mr)))

            # Normalo 2023: dm not known
            elif self.season.mm is None:
                solutions += [tenmatches.addpair(ap) for ap in addmatches_dict[mr]]

            # All other seasons
            else:
                if mr == self.season.mm:
                    solutions += [tenmatches.addpair(ap) for ap in addmatches_dict[mr]]
                else:
                    dmleft = [l for (l, r) in tenmatches if r == self.season.mm][0]
                    if Pair(dmleft, mr) not in addmatches_dict[mr]:
                        continue
                    solutions.append(tenmatches.addpair(Pair(dmleft, mr)))
            assert all(isinstance(x, Solution) for x in solutions)
        return solutions

    def generate_complete_solutions(self, sol: Solution, end: int) -> list[Solution]:
        """Generating solutions"""
        assert isinstance(sol, Solution)
        if len(sol) == self.season.nummatches and self.solution_possible(sol, end):
            return [sol]

        def zip_product(clefts, ordering):
            return frozenset(Pair(*p) for p in zip(clefts, ordering))

        _, _, dm_in_psol = self.get_solution_leftrights(sol)
        pos_matches = self.possible_matches_for_solution(sol, end)

        # for l in pos_matches:
        #     print(l, pos_matches[l])

        products = [
            list(ps)
            for ps in itertools.product(*pos_matches.values())
            if len(set(ps)) == len(ps)
        ]

        other_matches_list = list(
            map(lambda p: Solution(zip_product(pos_matches.keys(), p)), products)
        )

        if dm_in_psol > 0:
            # Multiple match is already in Solution
            return self.merge_mm_in_solution(sol, other_matches_list, end)

        solutions = self.merge_mm_not_in_solution(sol, other_matches_list, end)

        unique_sols = []

        for sol in solutions:
            assert (
                len(sol) == self.season.nummatches
            ), f"Complete solutions with {self.season.nummatches} pairs, not {len(sol)} pairs "
            _, rs = zip(*sol)

            assert (
                2 not in Counter(rs).values()
            ), f"No double matches for rights possible"

            if sol not in unique_sols:
                unique_sols.append(sol)
            assert all(isinstance(x, Pair) for x in sol)

        # if len(unique_sols) != len(solutions):
        #     print("423")

        return unique_sols

    def generate_partial_solutions(self, end: int) -> list[Solution]:

        nights = self.season.get_nights(end)
        kpm = self.season.get_pms(end)

        pairs_per_night = []
        for night in nights:
            notcorrect = list(
                filter(lambda p: self.no_match(p, end) or p in kpm, night.pairs)
            )
            defcorrect = list(filter(lambda p: p in kpm, night.pairs))
            remaining = set(night.pairs) - set(defcorrect) - set(notcorrect)
            assert all(isinstance(x, Pair) for x in kpm), list(kpm) + list(
                map(type, kpm)
            )

            combs = [
                Solution(frozenset(comb).union(kpm))
                for comb in itertools.combinations(
                    remaining, night.lights - len(defcorrect)
                )
            ]

            pairs_per_night.append(combs)

        night1, night2 = pairs_per_night[:2]
        self.merge_solutions_lists(night1, night2)

        merged_solutions: list[Solution] = functools.reduce(
            lambda g1, g2: self.merge_solutions_lists(g1, g2, end),
            pairs_per_night,
        )

        merged_solutions: list[Solution] = list(
            filter(lambda a: self.solution_possible(a, end), merged_solutions)
        )

        return merged_solutions
