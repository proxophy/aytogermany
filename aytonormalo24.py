from ayto import AYTO, analysize_solutions

from typing import Optional, Union
from collections import Counter
import itertools
import functools
import time
import pandas as pd
import utils

class AYTONormalo2024(AYTO):
    tm: str

    def __init__(self,
                 lefts: list[str], rights: list[str],
                 nights: list[tuple[list[tuple[str, str]], int]],
                 tm: str,
                 matchboxes: dict[tuple[str, str], bool] = {},
                 bonights: list[int] = [],
                 cancellednight: int = -1, boxesepisodes: list[int] = list(range(10)),
                 solution: Optional[set[tuple[str, str]]] = None) -> None:
        super().__init__(lefts, rights, nights, matchboxes, bonights,
                         None, None,  cancellednight, boxesepisodes, solution)
        self.tm = tm

    def no_match(self, l: str, r: str, options: dict[str, bool]) -> bool:
        nomatch = super().no_match(l, r,  options)

        kpm = self.get_pms(options)
        hpm = [e for p in kpm for e in p]

        if (l, r) not in kpm and ((l in hpm) or (r in hpm)):
            mmls = [p for p in hpm if Counter(hpm)[p] > 1]
            # Normalo 24
            if self.tm is not None and len(mmls) < 3 \
                    and l in mmls and r not in hpm:
                # we only know two out of three of the multiple matches in Normalo 2024 at episode 8
                return False
            return True

        return nomatch

    def generate_parsols(self,  options: dict) -> list[set[tuple[str, str]]]:
        # print("sub compute_guesses")
        return super().generate_parsols(options)

    def parsol_possible(self, guess: set[tuple[str, str]], options: dict) -> bool:
        pdict = {l: [] for l in self.lefts}
        for (l, r) in guess:
            pdict[l].append(r)
        g_lefts, g_rights = zip(*guess)

        # lefts with multiple matches
        multls = [l for l in self.lefts if len(pdict[l]) > 1]
        if len(multls) == 1:
            multl = multls[0]
            mutiplers = pdict[multl]
            if self.tm in g_rights and self.tm not in pdict[multl]:
                return False

            if len(mutiplers) == 3 and self.tm not in mutiplers:
                return False
        return super().parsol_possible(guess, options)

    def possible_matches_for_parsol(self, guess: set[tuple[str, str]], options: dict) -> dict[str, list[str]]:
        possible_matches = super().possible_matches_for_parsol(guess,  options)
        if len(guess) > 0:
            g_lefts, g_rights = zip(*guess)
            g_lefts, g_rights = list(g_lefts), list(g_rights)
        else:
            g_lefts, g_rights = {}, {}

        not_tm_in_asm = self.tm is not None and self.tm not in g_rights
        possible_matches = {
            l: [r
                for r in possible_matches[l]
                if (not (not_tm_in_asm) or r != self.tm)
                ]
            for l in possible_matches
        }

        return possible_matches

    def merge_mm_in_parsol(self, guess: set[tuple[str, str]],  others_list: list[set[tuple[str, str]]],
                                  options: dict):
        if len(guess) > 0:
            g_lefts, g_rights = zip(*guess)
            g_lefts, g_rights = list(g_lefts), list(g_rights)
        else:
            g_lefts, g_rights = {}, {}

        not_tm_in_asm = self.tm is not None and self.tm not in g_rights

        if len(g_lefts) - len(set(g_lefts)) == 2:
            if self.tm not in g_rights:
                print("tm must be part of tripple match")
                return []
            # print("TM IN ASM")
            return [guess.union(set(othermatches)) for othermatches in others_list]

        elif len(g_lefts) - len(set(g_lefts)) == 1:
            # print("DM OF TM IN ASM")
            counter = Counter(g_lefts)
            tmleft = [l for l in self.lefts if counter[l] == 2][0]
            solutions = []
            for othermatches in others_list:
                elevenmatches = guess.union(othermatches)
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

    def merge_mm_not_in_parsol(self, guess: set[tuple[str, str]], others_list: list[set[tuple[str, str]]], options: dict):
        solutions = []
        addmatches_dict = {r: [(l, r) for l in self.lefts if not self.no_match(l, r, options)]
                           for r in self.rights}

        for othermatches in others_list:
            tenmatches = guess.union(othermatches)
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


if __name__ == "__main__":

    season = utils.read_data("normalo2024")
