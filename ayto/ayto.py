import itertools
import functools
import time
from typing import Any

from .models import *


def double_match_for_right(sol: Solution):
    rs = []
    for _, r in sol:
        if r in rs:
            return True
        rs.append(r)
    return False


def get_candidates(sol: Solution):
    if len(sol) > 0:
        g_lefts, g_rights = zip(*sol)
        g_lefts, g_rights = list(g_lefts), list(g_rights)
    else:
        g_lefts, g_rights = set(), set()
    return set(g_lefts), set(g_rights), len(g_lefts) - len(set(g_lefts))


def mm_left(sol: Solution) -> str | None:
    ls = set()
    for l, _ in sol:
        if l in ls:
            return l
        ls.add(l)
    return None

def dict_rep(sol: Solution) -> dict[str, list[str]]:
    d = {}
    for l, r in sol:
        if l in d:
            d[l].append(r)
        else:
            d[l] = [r]
    return d

class Solver:
    season: Season
    sm = None

    def __init__(self, season: Season) -> None:
        self.season: Season = season
        self.no_match_precomputed = {
            ((l, r), end): self.no_match(l, r, end)
            for l in self.season.lefts
            for r in self.season.rights
            for end in range(0, 11)
        }
        self.reasons = []
        self.times = {
            "mm_in_sol": 0.0,
            "mm_not_in_sol": 0.0,
            "before_loop": 0.0,
            "before_if": 0.0,
            "mr2_if": 0.0,
            "mr2_else": 0.0,
            "mr2": 0.0,
            "mr1": 0.0,
            "making_set": 0.0,
            "until348": 0.0,
        }

    @functools.cache
    def no_match(self, l: str, r: str, end: int) -> bool:
        """(l,r) are definitely no match"""
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
            if not (
                self.season.name == "normalo2026"
                and end >= self.season.mm_known_after_week
            ):
                # special case normalo2026: DM (Noel, Alicia) known only after season
                # but also Alicia only known as mm after week 4
                return True

        return False

    def solution_correct_format(self, sol: Solution) -> dict:

        if len(sol) > self.season.num_matches:
            return {"res": False, "reason": "too_many_matches"}

        if double_match_for_right(sol):
            return {"res": False, "reason": "double_match_for_right"}
        for l, r in sol:
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
        if len(sol) > self.season.num_matches:
            return {"res": False, "reason": "too_many_matches"}
        end = max(0, min(end, 10))

        cf = self.solution_correct_format(sol)
        if not cf["res"]:
            return cf

        # no known no matches
        if any(
            self.no_match(*p, end) for p in sol
        ):  # any([self.no_match(p, end) for p in sol]):
            ba: list = [self.no_match(*p, end) for p in sol]
            trueindex = ba.index(True)
            p = list(sol)[trueindex]
            return {"res": False, "reason": f"known_no_match", "detail": p}

        complete: bool = len(sol) == self.season.num_matches
        # check conditions for double matches
        # if complete:
        double_match_logic = self.check_double_match_logic(sol, end)
        if not double_match_logic["res"]:
            return double_match_logic

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
            clights: int = len(sol & night.pairs)

            if clights > night.lights:
                return {"res": False, "reason": f"too_many_lights_in_night_{i}"}
            elif clights < night.lights:
                if complete:
                    return {"res": False, "reason": f"not_enough_lights_in_night_{i}"}
            i += 1
        return {"res": True, "reason": ""}

    def check_double_match_logic(self, sol: Solution, end: int) -> dict:
        if self.season.dmtuple is not None and end >= self.season.dmtupleknown:
            dml = [l for (l, r) in sol if r in self.season.dmtuple]
            if len(dml) > 1 and dml[0] != dml[1]:
                return {"res": False, "reason": "vip23_dmtuple_not_same_match"}

        # pdict = {l: [] for l in self.season.lefts}
        pdict = {}
        for l, r in sol:
            pdict.setdefault(l, []).append(r)
        multiplels = [l for l, rs in pdict.items() if len(rs) > 1]

        mm = self.season.mm if end >= self.season.mm_known_after_week else None

        if len(multiplels) == 2:
            if not self.season.two_dms:
                return {"res": False, "reason": "two_double_matches"}
            else:
                # VIP 2025: two double matches
                multiplers = [e for l in multiplels for e in pdict[l]]
                if not (end >= self.season.mm_known_after_week and mm in multiplers):
                    return {"res": False, "reason": "Jimi_not_in_double_match"}

        elif len(multiplels) == 1:
            mutiplel = multiplels[0]
            multiplers = pdict[mutiplel]

            if (
                (self.season.dmtuple is not None and end >= self.season.dmtupleknown)
                and len(multiplers) == 2
                and set(self.season.dmtuple) != set(multiplers)
            ):
                return {"res": False, "reason": "multiplelefts_not_dmtuple"}
            elif (
                mm is not None
                and end >= self.season.mm_known_after_week
                and len(multiplers) == self.season.max_multiple_match_size
                and mm not in pdict[mutiplel]
                and not self.season.two_dms
            ):
                return {
                    "res": False,
                    "reason": "mm_not_in_double_match",
                    "detail": (mutiplel, multiplers),
                }
        elif len(multiplels) > 2:
            return {"res": False, "reason": "too_many_double_matches"}

        return {"res": True, "reason": ""}

    def merge_solutions_lists(
        self, psl_1: list[Solution], psl_2: list[Solution], end: int, includenight: bool
    ):
        """
        Output: list of merged together partial sols
        """
        m_asm = set()
        for g1, g2 in itertools.product(psl_1, psl_2):
            g3 = g1.union(g2)
            pred = self.solution_possible(g3, end, includenight)
            if pred["res"]:
                m_asm.add(g3)
            else:
                self.reasons.append(pred["reason"])

        return list(m_asm)

    def possible_matches_for_solution(
        self, sol: Solution, end: int, includenight: bool
    ) -> dict[str, list[str]]:
        g_lefts, g_rights, _ = get_candidates(sol)

        nights = self.season.get_nights(end)
        if not includenight:
            nights = nights[:-1]
        sitting_nomatches = set()

        for night in nights:
            pl: int = len(sol & night.pairs)
            if pl < night.lights:
                continue
            for p in night.pairs - sol:
                sitting_nomatches.add(p)
        mm = self.season.mm if end >= self.season.mm_known_after_week else None
        pos_matches = {
            l: [
                r
                for r in set(self.season.rights) - g_rights
                if not self.no_match(l, r, end)
                and not (l, r) in sitting_nomatches
                and r != mm
            ]
            for l in self.season.lefts
            if l not in g_lefts
        }
        return pos_matches

    def merge_mm_in_solution(
        self, sol: Solution, other_matches_list: list[Solution], end: int
    ) -> list[Solution]:
        _, _, mm_num = get_candidates(sol)
        if mm_num == 0:
            raise ValueError(
                "merge_mm_in_solution shouldn't be called if muliple match(es) are not in Solution"
            )

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
        mm = self.season.mm if end >= self.season.mm_known_after_week else None
        sitting_nomatches: set[Pair] = set()
        # Consider sitting matches as no matches if not in Solution and we have the right
        # amount of lights
        for night in nights:
            pl: int = len(sol & night.pairs)
            if pl > night.lights:
                return []
            elif pl < night.lights:
                continue
            for p in night.pairs - sol:
                sitting_nomatches.add(p)

        solutions = set()
        addmatches_dict = {
            r: [
                (l, r)
                for l in self.season.lefts
                if not (
                    self.no_match(l, r, end)
                    or (l, r) in sitting_nomatches
                    or (self.sm and l == self.sm)
                )
            ]
            for r in self.season.rights
        }

        for othermatches in other_matches_list:
            tenmatches = sol.union(othermatches)

            _, crights = zip(*tenmatches)
            mr = [r for r in self.season.rights if r not in crights][0]  # missing right

            # VIP 2023: dmtuple
            if self.season.dmtuple is not None and end >= self.season.dmtupleknown:
                if mr not in self.season.dmtuple:
                    continue

                dmleft = [l for (l, r) in tenmatches if r in self.season.dmtuple][0]
                solutions.add(tenmatches | {(dmleft, mr)})

            # Normalo 2023/24/25%25: dm not known
            elif mm is None:
                solutions.update([tenmatches | {ap} for ap in addmatches_dict[mr]])

            # All other seasons
            else:
                if mr == mm:
                    solutions.update([tenmatches | {ap} for ap in addmatches_dict[mr]])
                else:
                    dmleft = [l for (l, r) in tenmatches if r == mm][0]
                    if (dmleft, mr) not in addmatches_dict[mr]:
                        continue
                    solutions.update([tenmatches | {(dmleft, mr)}])
        return list(solutions)

    def generate_complete_solutions(
        self, sol: Solution, end: int, includenight: bool
    ) -> list[Solution]:
        """Generating solutions"""
        if not self.solution_possible(sol, end, includenight)["res"]:
            return []
        if (
            len(sol) == self.season.num_matches
            and self.solution_possible(sol, end, includenight)["res"]
        ):
            return [sol]

        def zip_product(clefts, ordering) -> Solution:
            return Solution(p for p in zip(clefts, ordering))

        s = time.time()
        _, _, mm_num = get_candidates(sol)
        pos_matches = self.possible_matches_for_solution(sol, end, includenight)
        # for p in pos_matches:
        #     print(p, pos_matches[p])

        products = [
            list(ps)
            for ps in itertools.product(*pos_matches.values())
            if len(set(ps)) == len(ps)
        ]

        other_matches_list = list(
            map(lambda p: zip_product(pos_matches.keys(), p), products)
        )

        e = time.time()
        self.times["until348"] = e - s

        if (
            mm_num == 1
            and self.season.num_matches == 11
            or mm_num == 2
            and self.season.num_matches == 12
        ):
            # Multiple match is already in Solution
            s = time.time()
            sols = self.merge_mm_in_solution(sol, other_matches_list, end)
            e = time.time()
            self.times["mm_in_sol"] += e - s

            return sols

        s = time.time()
        solutions = self.merge_mm_not_in_solution(
            sol, other_matches_list, end, includenight
        )
        e = time.time()
        self.times["mm_not_in_sol"] += e - s

        s = time.time()
        unique_sols = set()
        for sol in solutions:
            unique_sols.add(sol)
        e = time.time()
        self.times["making_set"] += e - s

        return list(solutions)

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
                Solution(set(comb).union(kpm))
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

    def solve(self, end: int, includenight: bool) -> list[Solution]:
        s = time.time()
        partial_solutions = self.generate_partial_solutions(end, includenight)
        e = time.time()
        self.times["generate_partial_solutions"] = e - s
        s = time.time()

        solutions_unfiltered: set[Solution] = set()

        for g in partial_solutions:
            sols_g = self.generate_complete_solutions(g, end, includenight)
            solutions_unfiltered.update(sols_g)

        e = time.time()
        self.times["generate_complete_solutions"] = e - s
        s = time.time()

        # right amount of lights for every night, no no-matches and completeness are guaranteed
        # solutions = list(
        #     filter(
        #         lambda s: self.check_double_match_logic(s, end)["res"],
        #         solutions_unfiltered,
        #     )
        # )
        solutions = solutions_unfiltered

        e = time.time()
        self.times["check_double_match_logic"] = e - s
        s = time.time()

        return list(solutions)


# def compute_matches(
#     pos_matches: dict[str, list], lefts: list[str], rights: list[str], dm: str | None
# ):
#     def zip_product(clefts, ordering) -> Solution:
#         return Solution(p for p in zip(clefts, ordering))

#     if dm:
#         pos_matches_adjusted = {
#             l: {r for r in rs if not r == dm} for l, rs in pos_matches.items()
#         }
#     else:
#         pos_matches_adjusted = pos_matches

#     products = [
#         list(ps)
#         for ps in itertools.product(*pos_matches_adjusted.values())
#         if len(set(ps)) == len(ps)
#     ]
#     solutions = set()
#     graph_matchings = list(map(lambda p: zip_product(lefts, p), products))
#     if dm:
#         lefts_for_dm = [l for l in lefts if dm in pos_matches[l]]
#         for l in lefts_for_dm:
#             for matching in graph_matchings:
#                 solutions.add(matching | {(l, dm)})
#     else:
#         for matching in graph_matchings:
#             seated_rights = {r for _, r in matching}
#             missing_right = [r for r in rights if r not in seated_rights][0]
#             lefts_for_missing_right = [
#                 l for l in lefts if missing_right in pos_matches[l]
#             ]
#             for l in lefts_for_missing_right:
#                 solutions.add(matching | {(l, missing_right)})

#     return list(solutions)
