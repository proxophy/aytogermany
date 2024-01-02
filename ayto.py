from typing import List, Tuple, Dict, Set, Optional
from collections import Counter
import itertools
import functools
import time
import pandas as pd
import utils


class AYTO:

    lefts: List[str]
    rights: List[str]

    nights: List[Tuple[List[Tuple[str, str]], int]]
    matchboxes: Dict[Tuple[str, str], bool]
    bonights: List[int]

    dm: Optional[str]
    dmtuple: Optional[Tuple[str, str]]
    tm: Optional[str]

    numepisodes: int
    knownnights: List[int]
    knownboxes: List[int]
    knownpms: List[Tuple[str, str]]

    solution: Optional[Set[Tuple[str, str]]]

    def __init__(self,
                 lefts: List[str], rights: List[str],
                 nights: List[Tuple[List[Tuple[str, str]], int]],
                 matchboxes: Dict[Tuple[str, str], bool] = {},
                 bonights: List[int] = [],
                 dm: Optional[str] = None, dmtuple: Optional[Tuple[str, str]] = None,
                 tm: Optional[str] = None,
                 cancellednight: int = -1, boxesepisodes: List[int] = list(range(10)),
                 solution: Optional[Set[Tuple[str, str]]] = None) -> None:

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
        self.tm = tm

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

    def get_nights(self, end: int) -> List[Tuple[List[Tuple[str, str]], int]]:
        if end >= self.numepisodes - 1:
            return self.nights
        return self.nights[:(self.knownnights[end]+1)]

    def get_matchboxes(self, end: int) -> Dict[Tuple[str, str], bool]:
        if end >= self.numepisodes - 1:
            return self.matchboxes

        usedmbkeys = list(self.matchboxes.keys())[:(self.knownboxes[end]+1)]
        usedmb = {k: self.matchboxes[k] for k in usedmbkeys}
        return usedmb

    def get_bonights(self, end: int) -> List[int]:
        if end >= self.numepisodes - 1:
            return self.bonights
        return [bo for bo in self.bonights if bo <= self.knownnights[end]]

    def get_pms(self, end: int) -> List[Tuple[str, str]]:
        mb = self.get_matchboxes(end)
        return list(set(mb.keys()) & set(self.knownpms))

    def no_match(self, l: str, r: str, end: int) -> bool:
        '''(l,r) are definitely no match'''
        assert l in self.lefts and r in self.rights, f"f: {l}, s: {r}"

        nights = self.get_nights(end)
        mb = self.get_matchboxes(end)
        bon = self.get_bonights(end)
        kpm = self.get_pms(end)
        hpm = [e for p in kpm for e in p]

        # pair in blackout night who is not known perfect match
        if any([(l, r) in nights[bo][0] and (l, r) not in kpm for bo in bon]):
            return True

        # matchbox result was false
        if (l, r) in mb.keys() and not mb[(l, r)]:
            return True

        # one is part of known perfect match
        # consider multiple seated lefts
        if (l, r) not in kpm and ((l in hpm) or (r in hpm)):
            mmls = [p for p in hpm if Counter(hpm)[p] > 1]
            if self.tm is not None and len(mmls) < 3 \
                    and l in mmls and r not in hpm:
                # we only know two out of three of the multiple matches in Normalo 2024 at episode 8
                return False
            return True

        return False

    def guess_correct_format(self, guess: Set[Tuple[str, str]]) -> bool:
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

    def guess_possible(self, guess: Set[Tuple[str, str]], end, checknights: bool = True) -> bool:
        if len(guess) > self.nummatches:
            return False
        # assert len(guess) <= self.nummatches, \
        #     f"The guess has {len(guess)} instead of {self.nummatches}"

        if not self.guess_correct_format(guess):
            # print("Solution not correct format")
            # print(guess)
            return False

        complete: bool = len(guess) == self.nummatches

        kpm = self.get_pms(end)

        # no known no matches
        if any([self.no_match(*p, end) for p in guess]):
            ba: List = [self.no_match(*p, end) for p in guess]
            trueindex = ba.index(True)
            # print(f"guess has known no match: {list(guess)[trueindex]}")
            return False

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

        # cannot have more than 2/3 matches
        maxmatches = 3 if self.tm is not None else 2
        if any(len(pdict[l]) > maxmatches for l in pdict):
            return False

        # lefts with multiple matches
        mutiplels = [l for l in self.lefts if len(pdict[l]) > 1]
        if len(mutiplels) > 1:
            print("Only one double/tripple match")
            return False
        elif len(mutiplels) == 1:
            mutiplel = mutiplels[0]
            mutiplers = pdict[mutiplel]
            if self.tm in g_rights and self.tm not in pdict[mutiplel]:
                return False
            if self.tm is not None and len(mutiplers) == 3 and self.tm not in mutiplers:
                return False
            elif self.dmtuple is not None and len(mutiplers) == 2 and set(self.dmtuple) != set(mutiplers):
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
        nights = self.get_nights(end)
        for pairs, lights in nights:
            intersection = set(pairs) & guess
            clights = len(intersection)

            if clights > lights:
                return False
            elif clights < lights:
                if complete:
                    return False

        return True

    def merge_guesses_lists(self, gsl_1: List[Set], gsl_2: List[Set], end: int):
        '''
        Input: season, two list of guesses (set of pairs)
        Output: List of merged together guesses 
        '''

        m_asm = []
        for g1, g2 in itertools.product(gsl_1, gsl_2):
            pred = self.guess_possible(g1.union(g2), end, checknights=False)
            if pred and g1.union(g2) not in m_asm:
                m_asm.append(g1.union(g2))

        return m_asm

    def generate_solutions(self, guess: Set[Tuple[str, str]], end: int) -> List[Set[Tuple[str, str]]]:
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

        # is double/tripple match right in asm
        dm_in_asm = self.dm is not None and self.dm in g_rights
        not_tm_in_asm = self.tm is not None and self.tm not in g_rights
        dmt_in_asm = self.dmtuple is not None and (
            self.dmtuple[0] in g_rights or self.dmtuple[1] in g_rights)

        # possible matches for the lefts not assigned
        pos_matches = {l: [r for r in self.rights
                           if r not in g_rights and not self.no_match(l, r, end) and
                           (not (not_tm_in_asm) or r != self.tm) and
                           (not dm_in_asm or r != self.dm) and
                           (not dmt_in_asm or (self.dmtuple is not None and r not in self.dmtuple)) and
                           (dmt_in_asm or r != self.dm)]  # type: ignore
                       for l in self.lefts if l not in g_lefts}

        # for l in pos_matches:
        #     print(l, pos_matches[l])

        products = [list(ps) for ps in itertools.product(*pos_matches.values())
                    if len(set(ps)) == len(ps)]
        others_list = list(map(lambda p: zip_product(pos_matches.keys(), p),
                               products))

        # are the double/tripple matches in asm
        if self.tm is not None and len(g_lefts) - len(set(g_lefts)) == 2:
            if self.tm not in g_rights:
                print("tm must be part of tripple match")
                return []
            # print("TM IN ASM")
            return [guess.union(set(othermatches)) for othermatches in others_list]

        elif self.tm is not None and len(g_lefts) - len(set(g_lefts)) == 1:
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
                elif self.no_match(tmleft, missingright, end):
                    continue
                solutions.append(elevenmatches.union([(tmleft, missingright)]))

            return solutions

        elif self.tm is None and len(g_lefts) - len(set(g_lefts)) == 1:
            # print("DM IN ASM")
            return [guess.union(set(othermatches)) for othermatches in others_list]

        solutions = []

        addmatches_dict = {r: [(l, r) for l in self.lefts if not self.no_match(l, r, end)]
                           for r in self.rights}

        for othermatches in others_list:
            tenmatches = guess.union(othermatches)
            assert len(tenmatches) == 10

            _, crights = zip(*tenmatches)
            missingright = [r for r in self.rights if r not in crights][0]

            if self.tm is not None:
                missingrights = [r for r in self.rights if r not in crights]
                mr1, mr2 = missingrights
                if self.tm in missingrights:
                    # no multiple seatings, Mela not seated
                    # print("Mela not seated")
                    for l in self.lefts:
                        addmatches = [(l, mr1), (l, mr2)]
                        if all([not self.no_match(*p, end) for p in addmatches]):
                            solutions.append(tenmatches.union(addmatches))
                else:
                    # no mutiple seatings, Mela seated
                    # print("Mela seated")
                    dmleft = [l for (l, r) in tenmatches if r == self.tm][0]
                    addmatches = [(dmleft, mr1), (dmleft, mr2)]
                    if all([not self.no_match(*p, end) for p in addmatches]):
                        solutions.append(tenmatches.union(addmatches))
            elif self.dmtuple is not None:
                # VIP 2023
                if missingright not in self.dmtuple:
                    continue

                dmleft = [l for (l, r) in tenmatches if r in self.dmtuple][0]
                solutions.append(tenmatches.union([(dmleft, missingright)]))
            elif self.dm is not None:
                # Most other seasons
                if missingright == self.dm:
                    solutions += [tenmatches.union([(l, self.dm)])
                                  for l in self.lefts
                                  if not self.no_match(l, self.dm, end)]

                else:
                    dmleft = [
                        l for (l, r) in tenmatches if r == self.dm][0]

                    if self.no_match(dmleft, missingright, end):
                        continue
                    solutions.append(tenmatches.union(
                        [(dmleft, missingright)]))
            else:
                # Normalo 2023
                solutions += [tenmatches.union([ap])
                              for ap in addmatches_dict[missingright]]
                pass
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

    def compute_guesses(self, end: int) -> List[Set[Tuple[str, str]]]:
        nights = self.get_nights(end)
        kpm = self.get_pms(end)

        guesses_per_night = []
        for pairs, lights in nights:
            notcorrect = list(filter(lambda p: self.no_match(*p, end),
                                     pairs))
            defcorrect = list(filter(lambda p: p in kpm, pairs))
            remaining = set(pairs) - set(defcorrect) - set(notcorrect)

            combs = [set(comb).union(kpm)
                     for comb in itertools.combinations(remaining, lights - len(defcorrect))]

            guesses_per_night.append(combs)

        merged_guesses = functools.reduce(lambda g1, g2: self.merge_guesses_lists(g1, g2, end),
                                          guesses_per_night)

        merged_guesses = list(filter(lambda a: self.guess_possible(a, end),
                                     merged_guesses))

        guesses_lengths_counter = Counter(map(len, merged_guesses))
        print(f"Guesses lengths: {guesses_lengths_counter}")

        return merged_guesses


def find_solutions_slow(season: AYTO, end: int) -> List[Set[Tuple[str, str]]]:
    solutions = season.generate_solutions(set(), end)
    print(f"Generated solutions: {len(solutions)}")

    solutions = list(
        filter(lambda s: season.guess_possible(s, end), solutions))

    return solutions


def find_solutions(season: AYTO, end: int, asm: List = []) -> List[Set[Tuple[str, str]]]:
    merged_guesses = season.compute_guesses(end)

    solutions = []
    for g in merged_guesses:
        sols_g = season.generate_solutions(g, end)
        solutions += sols_g

    print("Generation done", len(solutions))
    # remove duplicates
    solutions = [set(item) for item in set(frozenset(item)
                                           for item in solutions)]
    print(f"Number of generated solutions: {len(solutions)}")
    solutions = list(filter(lambda s: season.guess_possible(s, end),
                            solutions))
    if len(asm) > 0:
        # keep solutions that have at least one pair of assumption
        solutions = [sol for sol in solutions
                     if any([p in sol for p in asm])]

    return solutions


def analysize_solutions(season: AYTO, end: int, asm: List = []) -> None:
    mbs = season.get_matchboxes(end)
    sols = find_solutions(season, end, asm)

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
               and not season.no_match(*p, end)]
    pms = [p for p in pdict if pdict[p] == len(sols)]
    newpms = [p for p in allpairs if p in pms and p not in mbs]
    data = {l: pd.Series([round(pdict.get((l, r), 0)/len(sols)*100, 1) for r in season.rights],
                         index=season.rights) for l in season.lefts}
    # data = {l: pd.Series([pdict.get((l, r), 0) for r in season.rights],
    #                      index=season.rights) for l in season.lefts}
    df = pd.DataFrame(data)

    print(f"Nach {end+2} Doppelfolgen")
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
                  "normalo2024"]

    for seasonfn in allseasons[7:]:
        print(seasonfn)
        season: AYTO = utils.read_data(seasonfn)

        end = 5
        assumption = []
        start = time.time()

        # sols = find_solutions(season, end, assumption)
        # print(len(sols))

        # sanity check for all past seasons
        # with open(f"analytics/{seasonfn}_half.txt", "r") as f:
        #     sols2 = [eval(l) for l in f.readlines()]
        #     sols2 = [s for s in sols2 if season.guess_possible(s, end)]
        # print(len(sols), len(sols2))

        analysize_solutions(season, end, assumption)
        # 2: 51701 - 39087
        # 3: 2456 - 2528

        print(time.time()-start, "s")
        print()
