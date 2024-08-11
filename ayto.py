from typing import Optional, Union
from pydantic import BaseModel
from collections import Counter
import itertools
import functools
import time
import pandas as pd
# import utils


class Solution (BaseModel):
    pairs: list[tuple[str, str]]


class AYTO:

    lefts: list[str]  # gender group of 10
    rights: list[str]  # gender group of 11

    nights: list[tuple[list, int]]
    matchboxes: dict[tuple[str, str], bool]
    bonights: list[int]

    dm: Optional[str]
    dmtuple: Optional[tuple[str, str]]

    numepisodes: int
    knownnights: list[int]
    knownboxes: list[int]
    knownpms: list[tuple[str, str]]

    solution: Optional[set[tuple[str, str]]]

    def __init__(self,
                 lefts: list[str], rights: list[str],
                 nights: list[tuple[list[tuple[str, str]], int]],
                 matchboxes: dict[tuple[str, str], bool] = {},
                 bonights: list[int] = [],
                 dm: Optional[str] = None, dmtuple: Optional[tuple[str, str]] = None,
                 cancellednight: int = -1, boxesepisodes: list[int] = list(range(10)),
                 solution: Optional[set[tuple[str, str]]] = None) -> None:

        self.count = 0

        self.lefts = lefts
        self.rights = rights
        self.nummatches = max(len(lefts), len(rights))
        if self.nummatches > 11:
            print("MORE THAN 11 MATCHES")

        self.nights = nights
        self.matchboxes = matchboxes
        self.bonights = bonights

        self.dm = dm
        self.dmtuple = dmtuple
        # self.tm = tm

        self.solution = solution
        # added for analyzing
        if cancellednight == -1:
            self.numepisodes = max(max(boxesepisodes)+1, len(self.nights))
            self.knownnights = [i for i in range(self.numepisodes)]
        else:
            self.numepisodes = max(max(boxesepisodes)+1, len(self.nights)+1)
            self.knownnights = []
            for i in range(self.numepisodes):
                if i < cancellednight:
                    self.knownnights.append(i)
                else:
                    self.knownnights.append(i-1)

        self.knownboxes = [0 for _ in range(self.numepisodes)]
        for i, e in enumerate(boxesepisodes):
            self.knownboxes[e] = i
        for i in range(1, len(self.knownboxes)):
            if self.knownboxes[i] == 0:
                self.knownboxes[i] = self.knownboxes[i - 1]

        # known perfect matches
        self.knownpms = [p for p in self.matchboxes.keys()
                         if self.matchboxes[p]]
        # people who have known perfect match
        self.haspm = [e for p in self.knownpms for e in p]

    def get_nights(self, options: dict) -> list[tuple[list, int]]:
        includenight: bool = options.get("includenight", True)
        end: int = options.get("end", self.numepisodes - 1)

        if end >= self.numepisodes - 1:
            nights = self.nights
        else:
            nights = self.nights[:(self.knownnights[end]+1)]
        return nights if includenight else nights[:-1]

    def get_matchboxes(self, options: dict) -> dict[tuple[str, str], bool]:
        end: int = options.get("end", self.numepisodes - 1)

        if end >= self.numepisodes - 1:
            return self.matchboxes

        usedmbkeys = list(self.matchboxes.keys())[:(self.knownboxes[end]+1)]
        usedmb = {k: self.matchboxes[k] for k in usedmbkeys}
        return usedmb

    def get_bonights(self, options: dict) -> list[int]:
        end: int = options.get("end", self.numepisodes - 1)
        if end >= self.numepisodes - 1:
            return self.bonights
        return [bo for bo in self.bonights if bo <= self.knownnights[end]]

    def get_pms(self,  options: dict) -> list[tuple[str, str]]:
        mb = self.get_matchboxes(options)
        return list(set(mb.keys()) & set(self.knownpms))

    def no_match(self, l: str, r: str,  options: dict[str, bool]) -> bool:
        '''(l,r) are definitely no match'''
        self.count += 1
        assert l in self.lefts and r in self.rights, f"l: {l}, r: {r}"

        nights = self.get_nights(options)
        mb = self.get_matchboxes(options)
        bon = self.get_bonights(options)
        kpm = self.get_pms(options)
        hpm = [e for p in kpm for e in p]

        # pair in blackout night who is not known perfect match
        if any([(l, r) in nights[bo][0] and (l, r) not in kpm for bo in bon]):
            return True

        # matchbox result was false
        if (l, r) in mb and not mb[(l, r)]:
            return True

        # one is part of known perfect match
        # consider multiple seated lefts
        if (l, r) not in kpm and ((l in hpm) or (r in hpm)):
            return True

        return False

    def guess_correct_format(self, guess: set[tuple[str, str]]) -> bool:
        if len(guess) > self.nummatches:
            print("len(solution) > self.nummatches")
            return False

        leftseated = set()
        rightseated = set()
        dml = None

        for l, r in guess:
            if l not in self.lefts or r not in self.rights:
                print(f"name of {l} or {r} is wrong",
                      l in self.lefts, r in self.rights)
                return False
            if l in leftseated:
                if dml is None:
                    dml = l
                elif l != dml:
                    # print(f"we cannot have more than one double match: {l} {dml}")
                    return False
            else:
                leftseated.add(l)

            if r in rightseated:
                # print("no double matches for rights possible")
                return False
            else:
                rightseated.add(r)

        return True

    def guess_possible(self, guess: set[tuple[str, str]],  options: dict) -> bool:
        if len(guess) > self.nummatches:
            return False
        # assert len(guess) <= self.nummatches, \
        #     f"The guess has {len(guess)} instead of {self.nummatches}"
        checknights: bool = options.get("checknights", False)

        if not self.guess_correct_format(guess):
            return False

        complete: bool = len(guess) == self.nummatches

        kpm = self.get_pms(options)

        # no known no matches
        if any([self.no_match(*p,  options) for p in guess]):
            ba: list = [self.no_match(*p,  options) for p in guess]
            trueindex = ba.index(True)
            print(f"guess has known no match: {list(guess)[trueindex]}")
            return False

        # VIP 23
        # perfect matches of dmtuple must be the same person
        elif self.dmtuple is not None:
            dml = [l for (l, r) in guess if r in self.dmtuple]
            if len(dml) > 1 and dml[0] != dml[1]:
                # print(f"dmtuple rights do not have same pm {len(guess)}")
                return False

        # check condition for double matches
        pdict = {l: [] for l in self.lefts}
        for (l, r) in guess:
            pdict[l].append(r)
        g_lefts, g_rights = zip(*guess)

        # lefts with multiple matches
        mutiplels = [l for l in self.lefts if len(pdict[l]) > 1]
        if len(mutiplels) > 1:
            print("Only one double/tripple match")
            return False
        elif len(mutiplels) == 1:
            mutiplel = mutiplels[0]
            mutiplers = pdict[mutiplel]

            if self.dmtuple is not None and len(mutiplers) == 2 and set(self.dmtuple) != set(mutiplers):
                return False
            elif self.dm is not None and len(mutiplers) == 2 and self.dm not in pdict[mutiplel]:
                return False

        # if guess has nummatches matches
        if complete:
            # we must have all perfect matches
            if not all([pm in guess for pm in kpm]):
                print("complete and not all perfect matches in guess")
                return False

            # we must have 10 seated lefts and nummatches rights
            g_lefts, g_rights = zip(*guess)
            if len(set(g_lefts)) != 10 or len(set(g_rights)) != self.nummatches:
                return False

        if not checknights:
            return True

        # number of lights with pairs matching lights in nights
        # if we don't have self.nummatches pairs, we allow lesser lights
        nights = self.get_nights(options)
        for pairs, lights in nights:
            intersection = set(pairs) & guess
            clights = len(intersection)

            if clights > lights:
                return False
            elif clights < lights:
                if complete:
                    return False

        return True

    def merge_guesses_lists(self, gsl_1: list[set], gsl_2: list[set],
                            options: dict):
        '''
        Input: season, two list of guesses (set of pairs)
        Output: list of merged together guesses 
        '''
        options.update({"checknights": False})

        m_asm = []
        for g1, g2 in itertools.product(gsl_1, gsl_2):
            pred = self.guess_possible(
                g1.union(g2), options)
            if pred and g1.union(g2) not in m_asm:
                m_asm.append(g1.union(g2))

        return m_asm

    def possible_matches_for_guess(self, guess: set[tuple[str, str]], options: dict) -> dict[str, list[str]]:
        if len(guess) > 0:
            g_lefts, g_rights = zip(*guess)
            g_lefts, g_rights = list(g_lefts), list(g_rights)
        else:
            g_lefts, g_rights = {}, {}

        dm_in_asm = self.dm is not None and self.dm in g_rights
        dmt_in_asm = self.dmtuple is not None and (
            self.dmtuple[0] in g_rights or self.dmtuple[1] in g_rights)

        nights = self.get_nights(options)
        sitting_nomatches = []
        # Consider sitting matches as no matches if not in guess
        for pairs, _ in nights:
            sitting_nomatches += set(pairs) - guess

        # TODO: clean up
        pos_matches = {l: [r for r in self.rights
                           if r not in g_rights and not self.no_match(l, r,  options)
                           and (l, r) not in sitting_nomatches and
                           (not dm_in_asm or r != self.dm) and
                           (not dmt_in_asm or (self.dmtuple is not None and r not in self.dmtuple)) and
                           (dmt_in_asm or r != self.dm)]  # type: ignore
                       for l in self.lefts if l not in g_lefts}
        return pos_matches

    def merge_solutions_mm_in_sol(self, guess: set[tuple[str, str]], others_list: list[set[tuple[str, str]]], options: dict):
        if len(guess) > 0:
            g_lefts, g_rights = zip(*guess)
            g_lefts, g_rights = list(g_lefts), list(g_rights)
        else:
            g_lefts, g_rights = {}, {}

        if len(g_lefts) - len(set(g_lefts)) == 1:
            # print("DM IN ASM")
            return [guess.union(set(othermatches)) for othermatches in others_list]
        return []

    def merge_solutions_mm_not_in_sol(self, guess: set[tuple[str, str]],  others_list: list[set[tuple[str, str]]], options: dict):
        solutions = []
        addmatches_dict = {r: [(l, r) for l in self.lefts if not self.no_match(l, r,  options)]
                           for r in self.rights}

        for othermatches in others_list:
            tenmatches = guess.union(othermatches)
            assert len(tenmatches) == 10

            _, crights = zip(*tenmatches)
            missingright = [r for r in self.rights if r not in crights][0]

            # VIP 2023: dmtuple
            if self.dmtuple is not None:
                if missingright not in self.dmtuple:
                    continue

                dmleft = [l for (l, r) in tenmatches if r in self.dmtuple][0]
                solutions.append(tenmatches.union([(dmleft, missingright)]))

            # Normalo 2023: dm not known
            elif self.dm is None:
                solutions += [tenmatches.union([ap])
                              for ap in addmatches_dict[missingright]]

            # All other seasons
            else:
                if missingright == self.dm:
                    solutions += [tenmatches.union([(l, self.dm)])
                                  for l in self.lefts
                                  if not self.no_match(l, self.dm,  options)]

                else:
                    dmleft = [
                        l for (l, r) in tenmatches if r == self.dm][0]

                    if self.no_match(dmleft, missingright, options):
                        continue
                    solutions.append(tenmatches.union(
                        [(dmleft, missingright)]))
        return solutions

    def generate_solutions(self, guess: set[tuple[str, str]], options: dict) -> list[set[tuple[str, str]]]:
        '''Generating solutions'''
        if len(guess) == self.nummatches:
            return [guess]

        def zip_product(clefts, ordering):
            return set(zip(clefts, ordering))

        if len(guess) > 0:
            g_lefts, g_rights = zip(*guess)
            g_lefts, g_rights = list(g_lefts), list(g_rights)
        else:
            g_lefts, g_rights = {}, {}

        nights = self.get_nights(options)
        sitting_nomatches = []
        for pairs, _ in nights:
            sitting_nomatches += set(pairs) - guess

        pos_matches = self.possible_matches_for_guess(guess, options)

        # for l in pos_matches:
        #     print(l, pos_matches[l])

        products = [list(ps) for ps in itertools.product(*pos_matches.values())
                    if len(set(ps)) == len(ps)]
        others_list = list(map(lambda p: zip_product(pos_matches.keys(), p),
                               products))

        if len(g_lefts) - len(set(g_lefts)) > 0:
            # print("DM in guess")
            return self.merge_solutions_mm_in_sol(guess, others_list, options)

        solutions = self.merge_solutions_mm_not_in_sol(
            guess, others_list, options)

        # print(len(solutions))
        solutions2 = []

        for sol in solutions:
            if sol not in solutions2:
                solutions2.append(sol)

            if len(sol) != self.nummatches:
                raise ValueError
            _, rs = zip(*sol)
            if 2 in Counter(rs).values():
                raise ValueError

        return solutions2

    def compute_guesses(self, options: dict) -> list[set[tuple[str, str]]]:
        verbose: bool = options["verbose"]

        if verbose:
            print("compute_guesses")

        nights = self.get_nights(options)
        kpm = self.get_pms(options)

        guesses_per_night = []
        for pairs, lights in nights:
            notcorrect = list(filter(lambda p: self.no_match(*p, options),   # type: ignore
                                     pairs))  # type: ignore
            defcorrect = list(filter(lambda p: p in kpm, pairs))
            remaining = set(pairs) - set(defcorrect) - set(notcorrect)

            combs = [set(comb).union(kpm)
                     for comb in itertools.combinations(remaining, lights - len(defcorrect))]

            guesses_per_night.append(combs)

        merged_guesses = functools.reduce(lambda g1, g2: self.merge_guesses_lists(g1, g2,  options),
                                          guesses_per_night)

        options.update({"checknights": True})
        merged_guesses = list(filter(lambda a: self.guess_possible(a, options),
                                     merged_guesses))

        guesses_lengths_counter = Counter(map(len, merged_guesses))
        if verbose:
            print(f"Guesses lengths: {guesses_lengths_counter}")

        return merged_guesses


def find_solutions_slow(season: AYTO, options: dict) -> list[set[tuple[str, str]]]:
    solutions = season.generate_solutions(set(),  options)
    print(f"Generated solutions: {len(solutions)}")
    options.update({"checknights": True})
    solutions = list(
        filter(lambda s: season.guess_possible(s, options), solutions))

    return solutions


def find_solutions(season: AYTO,  options: dict, asm: list = []) -> list[set[tuple[str, str]]]:
    start = time.time()
    verbose: bool = options["verbose"]

    merged_guesses = season.compute_guesses(options)
    if verbose:
        print("compute_guesses", time.time()-start)
        start = time.time()

    solutions = []
    for g in merged_guesses:
        sols_g = season.generate_solutions(g,  options)
        solutions += sols_g

    if verbose:
        print("generate_solution", time.time()-start)
        start = time.time()

    if verbose:
        print(f"Number of generated solutions: {len(solutions)}")

    options.update({"checknights": True})
    solutions = list(filter(lambda s: season.guess_possible(s, options),
                            solutions))
    if verbose:
        print("filtering", time.time()-start)
        start = time.time()

    if len(asm) > 0:
        # keep solutions that have at least one pair of assumption
        solutions = [sol for sol in solutions
                     if any([p in sol for p in asm])]

    return solutions


def analysize_solutions(season: AYTO, options: dict, asm: list = []) -> None:
    mbs = season.get_matchboxes(options)
    sols = find_solutions(season,  options, asm)
    end = options.get("end", season.numepisodes-1)

    pdict = {}
    for s in sols:
        for p in s:
            if p in pdict:
                pdict[p] += 1
            else:
                pdict[p] = 1

    pitems = sorted(pdict.items(), key=lambda x: x[1], reverse=True)

    allpairs = [(l, r) for l in season.lefts for r in season.rights]
    impairs = [p for p in allpairs if p not in pdict and p not in mbs
               and not season.no_match(*p,  options)]
    pms = [p for p in pdict if pdict[p] == len(sols)]
    newpms = [p for p in allpairs if p in pms and p not in mbs]
    data = {l: pd.Series([round(pdict.get((l, r), 0)/len(sols)*100, 1) for r in season.rights],
                         index=season.rights) for l in season.lefts}
    # data = {l: pd.Series([pdict.get((l, r), 0) for r in season.rights],
    #                      index=season.rights) for l in season.lefts}
    df = pd.DataFrame(data)

    print(f"Nach {end+1} Doppelfolgen")
    print(f"Anzahl Möglichkeiten: {len(sols)}")
    print(df)
    print(f"Bekannte Perfect Matches: {pms}")
    if len(newpms) > 0:
        print(f"Neue Perfect Matches: {newpms}")
    if len(impairs) > 0:
        impdict = {l: [] for l in season.lefts}
        for l, r in impairs:
            impdict[l].append(r)
        print("Neue No-Matches durch Ausschlussprinzip:")
        for l in impdict:
            if len(impdict[l]) == 0:
                continue
            print(f"{l}: {impdict[l]}")

    return


if __name__ == "__main__":
    allseasons = ["normalo2020", "normalo2021", "normalo2022", "normalo2023",
                  "vip2021", "vip2022", "vip2023",
                  "vip2024"]

    import utils
    from aytonormalo24 import AYTONormalo2024

    season: AYTO = AYTO(*utils.read_data("vip2022"))
    options = {"end": 3,
               "includenight": True,
               "verbose": True}

    sols = find_solutions(season,  options)
    print(len(sols))

    # seasonn2024: AYTONormalo2024 = AYTONormalo2024(*utils.read_data_normalo2024("normalo2024"))
    # sols = find_solutions(seasonn2024, end, options)

    # print(len(sols))
