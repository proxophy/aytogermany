import itertools
import functools

from .models import *


class Solver:
    season: Season
    sm = None

    def __init__(self, season: Season) -> None:
        self.season: Season = season
        self.mmknown = (
            5 if self.season.name == "normalo2026" else 0
        )  # after which episode we know who is the mm
        self.no_match_precomputed = {
            ((l, r), end): self.no_match(l, r, end)
            for l in self.season.lefts
            for r in self.season.rights
            for end in range(0, 11)
        }

    def no_match(self, l: str, r: str, end: int) -> bool:
        """(l,r) are definitely no match"""
        assert l in self.season.lefts and r in self.season.rights, f"l: {l}, r: {r}"
        end = max(0, min(end, 10))

        mb = self.season.get_matchboxes(end)
        kpm = self.season.get_pms(end)
        hpm = [e for pp in kpm for e in pp]

        # matchbox result was false
        if (l, r) in mb and not mb[(l, r)]:
            return True

        # one is part of known perfect match
        # consider multiple seated lefts
        # assumption: if dm is not known and you get double match,
        # you find the double match in the same episode
        if (l, r) not in kpm and (l in hpm or r in hpm):
            if not (self.season.name == "normalo2026" and end >= self.mmknown):
                # special case normalo2026: DM (Noel, Alicia) known only after season
                # but also Alicia only known as mm after week 4
                return True

        return False

    def solution_correct_format(self, sol: Solution) -> dict:

        if len(sol) > self.season.nummatches:
            return {"res": False, "reason": "too_many_matches"}

        if sol.double_match_for_right():
            return {"res": False, "reason": "double_match_for_right"}
        for l, r in sol.pairs:
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

    def solution_possible(self, sol: Solution, end: int, includenight: bool) -> dict:
        if len(sol) > self.season.nummatches:
            return {"res": False, "reason": "too_many_matches"}
        end = max(0, min(end, 10))

        cf = self.solution_correct_format(sol)
        if not cf["res"]:
            return cf

        complete: bool = len(sol) == self.season.nummatches

        # no known no matches
        if any(
            self.no_match_precomputed[p, end] for p in sol
        ):  # any([self.no_match(p, end) for p in sol]):
            ba: list = [self.no_match(*p, end) for p in sol]
            trueindex = ba.index(True)
            p = list(sol.pairs)[trueindex]
            return {"res": False, "reason": f"known_no_match", "detail": p}

        # VIP 23
        # perfect matches of dmtuple must be the same person
        if self.season.dmtuple is not None and end >= self.season.dmtupleknown:
            dml = [l for (l, r) in sol if r in self.season.dmtuple]
            if len(dml) > 1 and dml[0] != dml[1]:
                return {"res": False, "reason": "vip23_dmtuple_not_same_match"}

        # check condition for double matches
        pdict = {l: [] for l in self.season.lefts}
        for l, r in sol:
            pdict[l].append(r)
        g_lefts, g_rights = zip(*sol)

        # lefts with multiple matches
        mutiplels = [l for l in self.season.lefts if len(pdict[l]) > 1]
        mm = self.season.mm if end >= self.mmknown else None
        if len(mutiplels) == 2:
            if not self.season.two_dms:
                return {"res": False, "reason": "two_double_matches"}
            else:
                # VIP 2025: two double matches
                multiplers = [e for l in mutiplels for e in pdict[l]]
                if not (end >= self.mmknown and mm in multiplers):
                    return {"res": False, "reason": "Jimi_not_in_double_match"}
        elif len(mutiplels) == 1:
            mutiplel = mutiplels[0]
            multiplers = pdict[mutiplel]

            if (
                (self.season.dmtuple is not None and end >= self.season.dmtupleknown)
                and len(multiplers) == 2
                and set(self.season.dmtuple) != set(multiplers)
            ):
                return {"res": False, "reason": "multiplelefts_not_dmtuple"}
            elif (
                mm is not None
                and end >= self.mmknown
                and len(multiplers) == 2
                and mm not in pdict[mutiplel]
                and not self.season.two_dms
            ):
                return {"res": False, "reason": "mm_not_in_double_match"}
        elif len(mutiplels) > 2:
            return {"res": False, "reason": "too_many_double_matches"}

        # if Solution has nummatches matches
        if complete:
            # we must have 10 seated lefts and nummatches rights
            g_lefts, g_rights = zip(*sol)
            if len(set(g_lefts)) != len(self.season.lefts) or len(set(g_rights)) != len(
                self.season.rights
            ):
                return {
                    "res": False,
                    "reason": "not_all_candidates_in_complete_solution",
                }
        # number of lights with pairs matching lights in nights
        # if we don't have self.nummatches pairs, we allow lesser lights

        nights = self.season.get_nights(end)
        if not includenight:
            nights = nights[:-1]
        i = 0
        for night in nights:
            clights = sol.intersectionlength(night.pairs)

            if clights > night.lights:
                return {"res": False, "reason": f"too_many_lights_in_night_{i}"}
            elif clights < night.lights:
                if complete:
                    return {"res": False, "reason": f"not_enough_lights_in_night_{i}"}
            i += 1
        return {"res": True, "reason": ""}

    def merge_solutions_lists(
        self, psl_1: list[Solution], psl_2: list[Solution], end: int, includenight: bool
    ):
        """
        Output: list of merged together partial sols
        """
        m_asm = []
        for g1, g2 in itertools.product(psl_1, psl_2):
            pred = self.solution_possible(g1.union(g2), end, includenight)
            if pred["res"] and g1.union(g2) not in m_asm:
                m_asm.append(g1.union(g2))

        return m_asm

    def possible_matches_for_solution(
        self, sol: Solution, end: int, includenight: bool
    ) -> dict[str, list[str]]:
        g_lefts, g_rights, _ = sol.get_candidates()

        nights = self.season.get_nights(end)
        if not includenight:
            nights = nights[:-1]
        sitting_nomatches = {}
      
        for night in nights:
            pl = sol.intersectionlength(night.pairs)
            if pl < night.lights:
                continue
            for p in set(night.pairs) - set(sol.pairs):
                sitting_nomatches[p] = True
        mm = self.season.mm if end >= self.mmknown else None
        pos_matches = {
            l: [
                r
                for r in set(self.season.rights) - g_rights
                if not self.no_match(l, r, end)
                and not sitting_nomatches.get((l, r), False)
                and r != mm
            ]
            for l in self.season.lefts
            if l not in g_lefts
        }
        return pos_matches

    def merge_mm_in_solution(
        self, sol: Solution, other_matches_list: list[Solution], end: int
    ) -> list[Solution]:
        _, _, mm_num = sol.get_candidates()
        assert (
            mm_num == 1
        ), "merge_mm_in_Solution shouldn't be called if muliple match(es) are not in Solution"
        sols = [sol.union(othermatches) for othermatches in other_matches_list]
        return sols

    def merge_mm_not_in_solution(
        self,
        sol: Solution,
        other_matches_list: list[Solution],
        end: int,
        includenight: bool,
    ) -> list[Solution]:

        nights = self.season.get_nights(end)
        if not includenight:
            nights = nights[:-1]
        mm = self.season.mm if end >= self.mmknown else None
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
                (l, r)
                for l in self.season.lefts
                if not (
                    self.no_match(l, r, end)
                    or sitting_nomatches.get((l, r), False)
                    or (self.sm and l == self.sm)
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
                solutions.append(tenmatches.addpair((dmleft, mr)))

            # Normalo 2023/24/25%25: dm not known
            elif mm is None:
                solutions += [tenmatches.addpair(ap) for ap in addmatches_dict[mr]]

            # All other seasons
            else:
                if mr == mm:
                    solutions += [tenmatches.addpair(ap) for ap in addmatches_dict[mr]]
                else:
                    dmleft = [l for (l, r) in tenmatches if r == mm][0]
                    if (dmleft, mr) not in addmatches_dict[mr]:
                        continue
                    solutions.append(tenmatches.addpair((dmleft, mr)))
        return solutions

    def generate_complete_solutions(
        self, sol: Solution, end: int, includenight: bool
    ) -> list[Solution]:
        """Generating solutions"""
        if (
            len(sol) == self.season.nummatches
            and self.solution_possible(sol, end, includenight)["res"]
        ):
            return [sol]

        def zip_product(clefts, ordering):
            return frozenset(p for p in zip(clefts, ordering))

        _, _, mm_num = sol.get_candidates()
        pos_matches = self.possible_matches_for_solution(sol, end, includenight)

        products = [
            list(ps)
            for ps in itertools.product(*pos_matches.values())
            if len(set(ps)) == len(ps)
        ]

        other_matches_list = list(
            map(lambda p: Solution(zip_product(pos_matches.keys(), p)), products)
        )

        if (
            mm_num == 1
            and self.season.nummatches == 11
            or mm_num == 2
            and self.season.nummatches == 12
        ):
            # Multiple match is already in Solution
            sols = self.merge_mm_in_solution(sol, other_matches_list, end)
            return sols

        solutions = self.merge_mm_not_in_solution(
            sol, other_matches_list, end, includenight
        )

        unique_sols = set()

        for sol in solutions:
            assert (
                len(sol) == self.season.nummatches
            ), f"Complete solutions with {self.season.nummatches} pairs, not {len(sol)} pairs "

            assert (
                not sol.double_match_for_right()
            ), f"No double matches for rights possible"
            unique_sols.add(sol)

        return list(unique_sols)

    def generate_partial_solutions(
        self, end: int, includenight: bool
    ) -> list[Solution]:

        nights = self.season.get_nights(end)
        if not includenight:
            nights = nights[:-1]
        kpm = self.season.get_pms(end)

        pairs_per_night = []
        for night in nights:
            notcorrect = list(
                filter(lambda p: self.no_match(*p, end) or p in kpm, night.pairs)
            )
            defcorrect = list(filter(lambda p: p in kpm, night.pairs))
            remaining = set(night.pairs) - set(defcorrect) - set(notcorrect)

            combs = [
                Solution(frozenset(comb).union(kpm))
                for comb in itertools.combinations(
                    remaining, night.lights - len(defcorrect)
                )
            ]

            pairs_per_night.append(combs)

        merged_solutions: list[Solution] = functools.reduce(
            lambda g1, g2: self.merge_solutions_lists(g1, g2, end, includenight),
            pairs_per_night,
        )

        return merged_solutions

    def solve(self, end: int, includenight: bool)-> list[Solution]:
        merged_partialsols = self.generate_partial_solutions(end, includenight)
        
        solutions_unfiltered: list[Solution] = []
    
        for g in merged_partialsols:
            sols_g = self.generate_complete_solutions(g, end, includenight)
            solutions_unfiltered.extend(sols_g)
        
        solutions = list(
            filter(lambda s: self.solution_possible(s, end, includenight)["res"], solutions_unfiltered)
        )
        assert len(solutions_unfiltered) - len(solutions) == 0
    
        return solutions
