from ayto import AYTO

from typing import Optional, Union
from collections import Counter
import pandas as pd
import ayto.utils as utils

PartialSol = set[tuple[str, str]]


class AYTONormalo2024(AYTO):
    tm: str | None

    def __init__(
        self,
        lefts: list[str],
        rights: list[str],
        nights: list[tuple[list[tuple[str, str]], int]],
        matchboxes: dict[tuple[int, str, str], bool] = {},   
        tm: str | None = None,
        solution: Optional[set[tuple[str, str]]] = None,
    ) -> None:
        super().__init__(
            lefts, rights, nights, matchboxes, None, solution
        )
        self.tm = tm

    def no_match(self, l: str, r: str, options: dict[str, bool]) -> bool:
        nomatch = super().no_match(l, r, options)

        kpm = self.get_pms(options)
        hpm = [e for p in kpm for e in p]

        if (l, r) not in kpm and l in hpm:
            # lefts with multiple matches from matchboxes, appear x times
            # for x perfect matches
            mmls = [p for p in hpm if Counter(hpm)[p] > 1]
            # Normalo 24
            if len(mmls) < 3 and l in mmls and r not in hpm:
                # we only know two out of three of the multiple matches in Normalo 2024 at episode 10
                return False
            return True

        return nomatch

    def generate_partialsols(self, options: dict) -> list[set[tuple[str, str]]]:
        return super().generate_partialsols(options)

    def partialsol_possible(self, psol: PartialSol, options: dict) -> bool:
        pdict = {l: [] for l in self.lefts}
        tm_in_partialsol = False
        for l, r in psol:
            pdict[l].append(r)
            if r == self.tm:
                tm_in_partialsol = True

        # lefts with multiple matches
        multls = [l for l in self.lefts if len(pdict[l]) > 1]
        if len(multls) > 1:
            return False
        elif len(multls) == 1:
            multl = multls[0]
            multr = pdict[multl]
            # self.tm has to be one of multiple matches
            if tm_in_partialsol and (multl, self.tm) not in psol:
                return False
            elif len(multr) == 3 and self.tm not in multr:
                return False
        return super().partialsol_possible(psol, options)

    def possible_matches_for_partialsol(
        self, psol: PartialSol, options: dict
    ) -> dict[str, list[str]]:
        possible_matches = super().possible_matches_for_partialsol(psol, options)
        # Filter out tm
        possible_matches = {
            l: list(filter(lambda r: r != self.tm, possible_matches[l]))
            for l in possible_matches
        }
        return possible_matches

    def merge_mm_in_partialsol(
        self, psol: PartialSol, other_matches_list: list[PartialSol], options: dict
    ):
        if len(psol) > 0:
            g_lefts, g_rights = zip(*psol)
            g_lefts, g_rights = list(g_lefts), list(g_rights)
        else:
            g_lefts, g_rights = {}, {}

        not_tm_in_asm = self.tm is not None and self.tm not in g_rights

        if len(g_lefts) - len(set(g_lefts)) == 2:
            if self.tm not in g_rights:
                print("tm must be part of tripple match")
                return []
            # print("TM IN ASM")
            return [psol.union(set(othermatches)) for othermatches in other_matches_list]

        elif len(g_lefts) - len(set(g_lefts)) == 1:
            # print("DM OF TM IN ASM")
            counter = Counter(g_lefts)
            tmleft = [l for l in self.lefts if counter[l] == 2][0]
            solutions = []
            for othermatches in other_matches_list:
                elevenmatches = psol.union(othermatches)
                _, crights = zip(*elevenmatches)
                missingright = [r for r in self.rights if r not in crights][0]
                # if we still have to add Mela: skip when somebody else is missing
                if not_tm_in_asm and missingright != self.tm:
                    print("not_tm_in_asm and missingright != self.tm")
                    continue
                elif self.no_match(tmleft, missingright, options):
                    continue
                solutions.append(elevenmatches.union([(tmleft, missingright)]))

            return solutions
        return []

    def merge_mm_not_in_partialsol(
        self, psol: PartialSol, other_matches_list: list[PartialSol], options: dict
    ):
        solutions = []
        addmatches_dict = {
            r: [(l, r) for l in self.lefts if not self.no_match(l, r, options)]
            for r in self.rights
        }

        for othermatches in other_matches_list:
            tenmatches = psol.union(othermatches)
            assert len(tenmatches) == 10

            _, crights = zip(*tenmatches)
            missingright = [r for r in self.rights if r not in crights][0]

            # Normalo 2024
            missingrights = [r for r in self.rights if r not in crights]
            mr1, mr2 = missingrights
            if self.tm in missingrights:
                # no multiple seatings, Mela not seated
                # print("Mela not seated")
                for l in self.lefts:
                    addmatches = [(l, mr1), (l, mr2)]
                    if all([not self.no_match(*p, options) for p in addmatches]):
                        solutions.append(tenmatches.union(addmatches))
            else:
                # no mutiple seatings, Mela seated
                # print("Mela seated")
                dmleft = [l for (l, r) in tenmatches if r == self.tm][0]
                addmatches = [(dmleft, mr1), (dmleft, mr2)]
                if all([not self.no_match(*p, options) for p in addmatches]):
                    solutions.append(tenmatches.union(addmatches))

        return solutions

