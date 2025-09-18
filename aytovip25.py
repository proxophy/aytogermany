from ayto import AYTO, time_it

from typing import Optional, Union
from collections import Counter
import pandas as pd
import utils
import itertools

PartialSol = set[tuple[str, str]]
CompleteSol = set[tuple[str, str]]
Night = tuple[list[tuple[str, str]], int]
Matchboxes = dict[tuple[str, str], bool]


class AYTOVIP2025(AYTO):
    tm: str

    def __init__(self,
                 lefts: list[str], rights: list[str],
                 nights: list[tuple[list[tuple[str, str]], int]],
                 matchboxes: dict[tuple[str, str], bool] = {},
                 bonights: list[int] = [],
                 dm: str | None = None, dmtuple: tuple[str, str] | None = None,
                 cancellednight: int = -1, boxesepisodes: list[int] = list(range(10)),
                 solution: Optional[set[tuple[str, str]]] = None) -> None:
        super().__init__(lefts, rights, nights, matchboxes, bonights,
                         dm, None, cancellednight, boxesepisodes, solution)
        self.two_dms = True

    def no_match(self, l: str, r: str, options: dict[str, bool]) -> bool:
        return super().no_match(l, r,  options)

    def generate_parsols(self,  options: dict) -> list[set[tuple[str, str]]]:
        return super().generate_parsols(options)

    def parsol_possible(self, parsol: PartialSol, options: dict) -> bool:
        if len(parsol) == 0:
            return False
        p_lefts, p_rights = zip(*parsol)
        r_counter = Counter(p_lefts)
        # no tripple matches
        if any([v > 2 for v in r_counter.values()]):
            return False
        return super().parsol_possible(parsol, options)

    def possible_matches_for_parsol(self, guess: set[tuple[str, str]], options: dict) -> dict[str, list[str]]:
        return super().possible_matches_for_parsol(guess,  options)

    def merge_mm_in_parsol(self, parsol: PartialSol, others_list: list[PartialSol],
                           options: dict):
        return super().merge_mm_in_parsol(parsol, others_list, options)

    def merge_mm_not_in_parsol(self, parsol: PartialSol, other_matches_list: list[PartialSol], options: dict):
        solutions = []
        addmatches_dict = {r: [(l, r) for l in self.lefts if not self.no_match(l, r, options)]
                           for r in self.rights}
        for othermatches in other_matches_list:
            tenplusmatches = parsol.union(othermatches)
            _, crights = zip(*tenplusmatches)
            missingrights = [
                r for r in self.rights if r not in crights]

            if len(missingrights) == 2:
                # add two matches
                r1, r2 = missingrights
                if r1 == self.dm or r2 == self.dm:
                    for l1, _ in addmatches_dict[r1]:
                        for l2, _ in addmatches_dict[r2]:
                            if l1 == l2:
                                continue
                            solutions.append(
                                tenplusmatches.union({(l1, r1), (l2, r2)}))
                else:
                    dmleft = [
                        l for (l, r) in tenplusmatches if r == self.dm][0]
                    if not self.no_match(dmleft, r1, options):
                        solutions += [tenplusmatches.union([(dmleft, r1), ap])
                                      for ap in addmatches_dict[r2]
                                      if ap[0] != dmleft]
                    if not self.no_match(dmleft, r2, options):
                        solutions += [tenplusmatches.union([(dmleft, r2), ap])
                                      for ap in addmatches_dict[r1]
                                      if ap[0] != dmleft]
            else:
                mr = missingrights[0]
                if mr == self.dm:
                    solutions += [tenplusmatches.union([ap])
                                  for ap in addmatches_dict[mr]]
                else:
                    dmleft = [
                        l for (l, r) in tenplusmatches if r == self.dm][0]
                    if (dmleft, mr) not in addmatches_dict[mr]:
                        continue
                    solutions.append(tenplusmatches.union(
                        [(dmleft, mr)]))
        return solutions

    def get_parsol_leftrights(self, parsol: PartialSol):
        '''Is the multiple match in the partial sol'''
        if len(parsol) == 0:
            return set(), set(), False
        p_lefts, p_rights = zip(*parsol)
        multiplels = {l for l in p_lefts if Counter(p_lefts)[l] == 2}
        return set(p_lefts), set(p_rights), len(multiplels) == 2

    def generate_complete_solutions(self, psol: PartialSol, options: dict) -> list[CompleteSol]:
        '''Generating solutions'''
        return super().generate_complete_solutions(psol, options)


if __name__ == "__main__":

    season = utils.read_data("normalo2024")
