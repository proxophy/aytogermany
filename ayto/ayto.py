from collections import Counter
import itertools
import functools

from .models import *


class AYTO:

    lefts: list[str]  # gender group of 10
    rights: list[str]  # gender group of 11

    nights: list[Night]
    matchboxes: Matchboxes
    boxesepisodes: list[int]

    dm: str | None
    dmtuple: tuple[str, str] | None

    knownboxes: list[int]
    knownpms: list[Pair]

    solution: Solution | None

    def __init__(
        self,
        lefts: list[str],
        rights: list[str],
        nights: list[Night],
        matchboxes: Matchboxes = Matchboxes(),
        dm: str | None = None,
        solution: Solution | None = None,
    ) -> None:

        self.count = 0

        self.lefts = lefts
        self.rights = rights
        self.nummatches = max(len(lefts), len(rights))
        if self.nummatches > 11:
            print("MORE THAN 11 MATCHES")

        self.nights = nights
        self.matchboxes = (
            matchboxes  # {(l, r): enmatchboxes[(e, l, r)] for e, l, r in enmatchboxes}
        )

        self.dm = dm
        self.dmtuple = None
        self.dmtupleknown = 7
        self.two_dms = False
        # self.tm = tm

        self.solution = solution
        # added for analyzing
        self.boxesepisodes = [e for e, _, _ in matchboxes]
        self.numepisodes = max(max(self.boxesepisodes) + 1, len(self.nights))

        self.knownboxes = [0 for _ in range(self.numepisodes)]
        for i, e in enumerate(self.boxesepisodes):
            self.knownboxes[e] = i
        for i in range(1, len(self.knownboxes)):
            # if no matchboxes for this episode, we take the last known matchbox
            if self.knownboxes[i] == 0:
                self.knownboxes[i] = self.knownboxes[i - 1]

    def get_nights(self, options: dict) -> list[Night]:
        """Retrieve nights based on episode limits."""
        end = min(self.numepisodes - 1, options.get("end", self.numepisodes - 1))
        nights = self.nights[: end + 1]
        return nights if options.get("includenight", True) else nights[:-1]

    def get_matchboxes(self, options: dict) -> Matchboxes:
        # change this to match Maxbox class
        end = min(self.numepisodes - 1, options.get("end", self.numepisodes - 1))
        # print("end", end, self.matchboxes.get_matchboxes_until_episode(end))
        # usedmbkeys = list(self.matchboxes.keys())[: (self.knownboxes[end] + 1)]
        # usedmb = {k: self.matchboxes[k] for k in usedmbkeys}
        return self.matchboxes.get_matchboxes_until_episode(end)

    def get_pms(self, options: dict) -> list[Pair]:
        end = min(self.numepisodes - 1, options.get("end", self.numepisodes - 1))
        return self.matchboxes.get_perfect_matches(end)

    def no_match(self, p: Pair, options: dict[str, bool]) -> bool:
        """(l,r) are definitely no match"""
        self.count += 1
        assert isinstance(p, Pair)
        assert p.l in self.lefts and p.r in self.rights, f"l: {p.l}, r: {p.r}"

        nights = self.get_nights(options)
        mb = self.get_matchboxes(options)
        kpm = self.get_pms(options)
        hpm = [e for pp in kpm for e in pp]

        # pair in blackout night who is not known perfect match
        # todo: reimplement blackout pairs

        # matchbox result was false
        if p in mb and not mb[p]:
            return True

        # one is part of known perfect match
        # consider multiple seated lefts
        # assumption: if dm is not known and you get double match,
        # you find the double match in the same episode
        if p not in kpm and (p.l in hpm or p.r in hpm) and p.r != self.dm:
            return True

        return False

    def solution_correct_format(self, sol: Solution) -> bool:
        assert isinstance(sol, Solution)

        if len(sol) > self.nummatches:
            print("len(solution) > self.nummatches")
            return False

        ls, rs = zip(*sol)
        if 2 in Counter(rs).values():
            # print(f"No double matches for a right")
            return False
        if any(l not in self.lefts for l in ls):
            print(f"Wrongly written names: {[c for c in ls if c not in self.lefts]}")
            return False
        if any(r not in self.rights for r in rs):
            print(f"Wrongly written names: {[c for c in rs if c not in self.rights]}")
            return False

        return True

    def solution_possible(self, sol: Solution, options: dict) -> bool:
        assert isinstance(sol, Solution)
        if len(sol) > self.nummatches:
            return False
        # assert len(psol) <= self.nummatches, \
        #     f"The partial solution has {len(psol)} instead of {self.nummatches}"
        checknights: bool = options.get("checknights", True)
        end: int = options.get("end", self.numepisodes - 1)

        if not self.solution_correct_format(sol):
            return False

        complete: bool = len(sol) == self.nummatches

        kpm = self.get_pms(options)

        # no known no matches
        if any([self.no_match(p, options) for p in sol]):
            ba: list = [self.no_match(p, options) for p in sol]
            trueindex = ba.index(True)
            print(f"Solution has known no match: {list(sol)[trueindex]} {options}")
            return False

        # VIP 23
        # perfect matches of dmtuple must be the same person
        if self.dmtuple is not None and end >= self.dmtupleknown:
            dml = [l for (l, r) in sol if r in self.dmtuple]
            if len(dml) > 1 and dml[0] != dml[1]:
                print(f"dmtuple rights do not have same pm {len(sol)} {dml}")
                return False

        # check condition for double matches
        pdict = {l: [] for l in self.lefts}
        for l, r in sol:
            pdict[l].append(r)
        g_lefts, g_rights = zip(*sol)

        # lefts with multiple matches
        mutiplels = [l for l in self.lefts if len(pdict[l]) > 1]
        if len(mutiplels) == 2:
            if not self.two_dms:
                # print("Only one double/tripple match")
                return False
            else:
                # VIP 2025: two double matches
                multiplers = [e for l in mutiplels for e in pdict[l]]
                if not self.dm in multiplers:
                    return False
        elif len(mutiplels) == 1:
            mutiplel = mutiplels[0]
            multiplers = pdict[mutiplel]

            if (
                (self.dmtuple is not None and end >= self.dmtupleknown)
                and len(multiplers) == 2
                and set(self.dmtuple) != set(multiplers)
            ):
                return False
            elif (
                self.dm is not None
                and len(multiplers) == 2
                and self.dm not in pdict[mutiplel]
                and not self.two_dms
            ):
                return False
        elif len(mutiplels) > 2:
            return False

        # if Solution has nummatches matches
        if complete:
            # we must have 10 seated lefts and nummatches rights
            g_lefts, g_rights = zip(*sol)
            if len(set(g_lefts)) != len(self.lefts) or len(set(g_rights)) != len(
                self.rights
            ):
                return False

        if not checknights:
            return True
        # number of lights with pairs matching lights in nights
        # if we don't have self.nummatches pairs, we allow lesser lights
        nights = self.get_nights(options)
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

    def merge_Solutions_lists(
        self, psl_1: list[Solution], psl_2: list[Solution], options: dict
    ):
        """
        Input: season, two list of partial sols (set of pairs)
        Output: list of merged together partial sols
        """
        m_asm = []
        for g1, g2 in itertools.product(psl_1, psl_2):
            pred = self.solution_possible(g1.union(g2), options)
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
        self, sol: Solution, options: dict
    ) -> dict[str, list[str]]:
        assert isinstance(sol, Solution)
        g_lefts, g_rights, _ = self.get_solution_leftrights(sol)

        nights = self.get_nights(options)
        sitting_nomatches = {}
        # Consider sitting matches as no matches if not in partial sol
        for night in nights:
            for p in set(night.pairs) - set(sol.pairs):
                sitting_nomatches[p] = True

        pos_matches = {
            l: [
                r
                for r in set(self.rights) - g_rights
                if not self.no_match(Pair(l, r), options)
                and not sitting_nomatches.get((l, r), False)
                and r != self.dm
            ]
            for l in self.lefts
            if l not in g_lefts
        }
        return pos_matches

    def merge_mm_in_solution(
        self, sol: Solution, other_matches_list: list[Solution], options: dict
    ) -> list[Solution]:
        assert isinstance(sol, Solution)
        _, _, dm_in_psol = self.get_solution_leftrights(sol)
        assert (
            dm_in_psol
        ), "merge_mm_in_Solution shouldn't be called if muliple match(es) are not in Solution"
        sols = [sol.union(othermatches) for othermatches in other_matches_list]
        return sols

    def merge_mm_not_in_solution(
        self, sol: Solution, other_matches_list: list[Solution], options: dict
    ) -> list[Solution]:
        assert isinstance(sol, Solution)
        end: int = options.get("end", self.numepisodes - 1)

        nights = self.get_nights(options)
        sitting_nomatches = {}
        # Consider sitting matches as no matches if not in Solution and we have the right
        # amount of lights
        for night in nights:
            pl = sol.intersectionlength(night.pairs)
            if pl > night.lights:
                print("line 330")
                return []
            elif pl < night.lights:
                continue
            for p in set(night.pairs) - set(sol.pairs):
                sitting_nomatches[p] = True

        solutions = []
        addmatches_dict = {
            r: [
                Pair(l, r)
                for l in self.lefts
                if not (
                    self.no_match(Pair(l, r), options)
                    or sitting_nomatches.get((l, r), False)
                )
            ]
            for r in self.rights
        }

        for othermatches in other_matches_list:
            tenmatches = sol.union(othermatches)
            assert len(tenmatches) == 10

            _, crights = zip(*tenmatches)
            mr = [r for r in self.rights if r not in crights][0]  # missing right

            # VIP 2023: dmtuple
            if self.dmtuple is not None and end >= self.dmtupleknown:
                if mr not in self.dmtuple:
                    continue

                dmleft = [l for (l, r) in tenmatches if r in self.dmtuple][0]
                solutions.append(tenmatches.addpair(Pair(dmleft, mr)))

            # Normalo 2023: dm not known
            elif self.dm is None:
                solutions += [tenmatches.addpair(ap) for ap in addmatches_dict[mr]]

            # All other seasons
            else:
                if mr == self.dm:
                    solutions += [tenmatches.addpair(ap) for ap in addmatches_dict[mr]]
                else:
                    dmleft = [l for (l, r) in tenmatches if r == self.dm][0]
                    if Pair(dmleft, mr) not in addmatches_dict[mr]:
                        # print("continue here?", (dmleft, mr))
                        continue
                    solutions.append(tenmatches.addpair(Pair(dmleft, mr)))
            assert all(isinstance(x, Solution) for x in solutions)
        return solutions

    def generate_complete_solutions(
        self, sol: Solution, options: dict
    ) -> list[Solution]:
        """Generating solutions"""
        assert isinstance(sol, Solution)
        if len(sol) == self.nummatches and self.solution_possible(sol, options):
            return [sol]

        def zip_product(clefts, ordering):
            return set(Pair(*p) for p in zip(clefts, ordering))

        _, _, dm_in_psol = self.get_solution_leftrights(sol)
        pos_matches = self.possible_matches_for_solution(sol, options)

        # for l in pos_matches:
        #     print(l, pos_matches[l])

        # todo: maybe optimize this
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
            return self.merge_mm_in_solution(sol, other_matches_list, options)

        solutions = self.merge_mm_not_in_solution(sol, other_matches_list, options)

        unique_sols = []

        for sol in solutions:
            assert (
                len(sol) == self.nummatches
            ), f"Complete solutions with {self.nummatches} pairs, not {len(sol)} pairs "
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

    def generate_partial_solutions(self, options: dict) -> list[Solution]:
        verbose: bool = options["verbose"]

        if verbose:
            print("generate_partial_solutions")

        nights = self.get_nights(options)
        kpm = self.get_pms(options)

        Solutions_per_night = []
        for night in nights:
            notcorrect = list(
                filter(lambda p: self.no_match(p, options) or p in kpm, night.pairs)
            )
            defcorrect = list(filter(lambda p: p in kpm, night.pairs))
            remaining = set(night.pairs) - set(defcorrect) - set(notcorrect)
            assert all(isinstance(x, Pair) for x in kpm), list(kpm) + list(
                map(type, kpm)
            )

            combs = [
                Solution(set(comb).union(kpm))
                for comb in itertools.combinations(
                    remaining, night.lights - len(defcorrect)
                )
            ]

            Solutions_per_night.append(combs)

        merged_Solutions: list[Solution] = functools.reduce(
            lambda g1, g2: self.merge_Solutions_lists(g1, g2, options),
            Solutions_per_night,
        )

        merged_Solutions: list[Solution] = list(
            filter(lambda a: self.solution_possible(a, options), merged_Solutions)
        )

        if verbose:
            Solutions_lengths_counter = Counter(list(map(len, merged_Solutions)))
            print(f"Parsols lengths: {Solutions_lengths_counter}")

        return merged_Solutions


class SolutionSpace:
    sols: list[Solution]


def dm_left(sol: Solution) -> str:
    """Which of lefts has double match"""
    lefts = list(zip(*sol))[0]
    return [l for (l, r) in Counter(lefts).items() if r >= 2][0]
