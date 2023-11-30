from typing import List, Tuple, Dict, Set, Union, Optional
from collections import Counter
import itertools
import functools
import json
import pandas as pd



class AYTO:

    lefts: List[str]
    rights: List[str]

    nights: List[Tuple[List[Tuple[str, str]], int]]
    matchboxes: Dict[Tuple[str, str], bool]
    bonights: List[int]

    dm: Optional[str]
    dmtuple: Optional[Tuple[str, str]]

    cancellednight: int
    numepisodes: int
    knownnights: List[int]
    knownboxes: List[int]
    knownpm: List[Tuple[str, str]]
    knowhaspm: List[str]

    solution: Optional[Set[Tuple[str, str]]]

    def __init__(self,
                 fn: str, nightsfromexcel: bool = True) -> None:
        self.fn = fn
        self.read_data(fn, nightsfromexcel)

    def read_data(self, fn: str, nightsfromexcel: bool = True) -> None:
        def read_nights(nightsfromexcel: bool) -> None:
            '''Read nights and check if everything is fine'''

            if nightsfromexcel:
                print("NIGHTS FROM EXCEL")
                nightsdf = pd.read_excel(
                    f"data/{self.fn}.xlsx", sheet_name="Nights",
                    header=0, index_col=0)
                leftsh = nightsdf.columns.values.tolist()

                nights = [(list(zip(leftsh, row[:-1])), row[-1])
                          for row in nightsdf.values.tolist()]
            else:
                print("NIGHTS FROM JSON")
                # assert that nights are in data
                assert "nights" in jsondata, "data has not a night key"
                nights = [([tuple(p) for p in pairs], lights)
                          for pairs, lights in jsondata["nights"]]

            self.nights = []

            # make sure all pairs are valid pairs and there are no duplicates
            for pairs, lights in nights:
                seated_pairs = []

                seated_lefts, seated_rights = [], []

                for l, r in pairs:
                    if l in self.lefts and r in self.rights:
                        # check that we don't have double seatings
                        if l in seated_lefts or r in seated_rights:
                            raise ValueError(
                                f"{l} or {r} has already been seated")
                        seated_lefts.append(l)
                        seated_rights.append(r)

                        seated_pairs.append((l, r))
                    elif l in self.rights and r in self.lefts:
                        # check that we don't have double seatings
                        if l in seated_rights or r in seated_lefts:
                            raise ValueError(
                                f"{l} or {r} has already been seated")
                        seated_lefts.append(r)
                        seated_rights.append(l)

                        seated_pairs.append((r, l))
                    else:
                        raise ValueError(
                            f"{l} or {r} is neither in the list of women or men")

                self.nights.append((seated_pairs, lights))
            return

        def read_matchboxes() -> None:
            '''Read matchboxes'''
            assert "matchboxes" in jsondata, "data has not a night key"

            self.matchboxes = {}

            for (l, r), result in jsondata["matchboxes"]:
                if l in self.lefts and r in self.rights:
                    if (l, r) in self.matchboxes:
                        raise ValueError(f"Double matchbox result for {l,r}")
                    self.matchboxes[(l, r)] = result
                elif l in self.rights and r in self.lefts:
                    if (r, l) in self.matchboxes:
                        raise ValueError(f"Double matchbox result for {r,l}")
                    self.matchboxes[(r, l)] = result

        def read_boxesepisode() -> None:
            if "boxesepisode" in jsondata:
                assert len(self.matchboxes) == len(jsondata["boxesepisode"]), \
                    r"not every matchbox could be assigned a episode"

                self.knownboxes = [0 for _ in range(self.numepisodes)]
                for i, e in enumerate(jsondata["boxesepisode"]):
                    self.knownboxes[e] = i
                for i in range(1, len(self.knownboxes)):
                    if self.knownboxes[i] == 0:
                        self.knownboxes[i] = self.knownboxes[i - 1]
            else:
                self.knownboxes = [i for i in range(len(self.nights))]

        with open(f"data/{fn}.json", "r") as f:
            jsondata = json.loads(f.read())

        # set left and right
        if len(jsondata["women"]) == 11 and len(jsondata["men"]) == 10:
            self.lefts, self.rights = jsondata["men"], jsondata["women"]
        elif len(jsondata["women"]) == 10 and len(jsondata["men"]) == 11:
            self.lefts, self.rights = jsondata["women"], jsondata["men"]
        elif len(jsondata["men"]) == 10 and len(jsondata["women"]) == 12:
            print("Normalo 2024: 12 women")
            self.lefts, self.rights = jsondata["men"], jsondata["women"]
        else:
            raise ValueError(f"not enough or too much women or men")

        # seatings and lights for nights
        read_nights(nightsfromexcel)

        # matchboxes with results
        read_matchboxes()

        # nights with blackout
        self.bonights = jsondata["bonights"]

        # do we know who is the second match to someone
        # besides normalo2023, yes
        if "dm" in jsondata:
            self.dm = jsondata["dm"]
        else:
            self.dm = None
            print("dm is not known")
            # self.dm_rights = [r for r in self.rights if r not in self.haspm]

        # do we know which two share the same match
        # besides vip2023, no
        if "dmtuple" in jsondata:
            dm1, dm2 = jsondata["dmtuple"]
            if not (dm1 in self.rights and dm2 in self.rights):
                raise ValueError(
                    f"{dm1} or {dm2} may be misspelled for dmtuple")
            self.dmtuple = (dm1, dm2)
        else:
            self.dmtuple = None

        # added to do some funny analyzing
        if "cancelled" in jsondata:
            self.cancellednight = jsondata["cancelled"]
            self.numepisodes = len(self.nights) + 1
            self.knownnights = []
            for i in range(self.numepisodes):
                if i < self.cancellednight:
                    self.knownnights.append(i)
                else:
                    self.knownnights.append(i-1)
        else:
            self.cancellednight = -1
            self.numepisodes = len(self.nights)
            self.knownnights = [i for i in range(self.numepisodes)]

        read_boxesepisode()

        # known perfect matches
        self.knownpm = [p for p in self.matchboxes.keys()
                        if self.matchboxes[p]]
        # people who have known perfect match
        self.haspm = [e for p in self.knownpm for e in p]

        if "solution" in jsondata:
            self.solution = {(l, r) for l, r in jsondata["solution"]}
            print(self.solution)

    def write_nights_excel(self) -> None:
        dflist = []

        for pairs, lights in self.nights:
            ndict = {}
            for l, r in pairs:
                ndict[l] = r
            ndict["lights"] = lights
            dflist.append(ndict)

        df = pd.DataFrame(dflist)
        if len(self.nights) == 0:
            df = pd.DataFrame(columns=self.lefts + ["Lights"])

        df.to_excel(f"data/{self.fn}.xlsx",
                    sheet_name="Nights", merge_cells=False)

        return

    def nights_up_to_episode(self, end: int) -> List[Tuple[List[Tuple[str, str]], int]]:
        if end >= self.numepisodes - 1:
            return self.nights
        return self.nights[:(self.knownnights[end]+1)]

    def matchboxes_up_to_episode(self, end: int) -> Dict[Tuple[str, str], bool]:
        if end >= self.numepisodes - 1:
            return self.matchboxes

        usedmbkeys = list(self.matchboxes.keys())[:(self.knownboxes[end]+1)]
        usedmb = {k: self.matchboxes[k] for k in usedmbkeys}
        return usedmb

    def bonights_up_to_episode(self, end: int) -> List[int]:
        if end >= self.numepisodes - 1:
            return self.bonights
        return [bo for bo in self.bonights if bo <= self.knownnights[end]]

    def knownpm_up_to_episode(self, end: int) -> List[Tuple[str, str]]:
        mb = self.matchboxes_up_to_episode(end)
        return list(set(mb.keys()) & set(self.knownpm))

    def get_possible_matches(self, end: int) -> Dict[str, List[str]]:
        # mb, nights, kpm, hpm, bon = self.get_upto_episode(end)
        possible_matches = {l: [r for r in self.rights if not self.no_match(l, r, end)]
                            for l in self.lefts}
        # change possible matches
        if self.dmtuple is not None:
            dm1, dm2 = self.dmtuple
            for l in possible_matches:
                # if not still both possible for this left, remove them as an option
                if not (dm1 in possible_matches[l] and dm2 in possible_matches[l]):
                    possible_matches[l] = list(
                        set(possible_matches[l]) - {dm1, dm2})

        return possible_matches

    def no_match(self, l: str, r: str, end: int) -> bool:
        '''(l,r) are definitely no match'''
        assert l in self.lefts and r in self.rights, f"f: {l}, s: {r}"

        nights = self.nights_up_to_episode(end)
        mb = self.matchboxes_up_to_episode(end)
        bon = self.bonights_up_to_episode(end)
        kpm = self.knownpm_up_to_episode(end)
        hpm = [e for p in kpm for e in p]

        # one is part of known perfect match
        # matchbox result was false
        # pair in blackout night who is not known perfect match
        return ((l, r) not in kpm and ((l in hpm) or (r in hpm))) \
            or ((l, r) in mb.keys() and not mb[(l, r)]) \
            or any([(l, r) in nights[bo][0] and (l, r) not in kpm for bo in bon])

    def guess_correct_format(self, guess: Set[Tuple[str, str]]) -> bool:
        if len(guess) > 11:
            print("len(solution) > 11")
            return False

        leftseated = set()
        rightseated = set()
        dml = None

        for l, r in guess:
            if l not in self.lefts or r not in self.rights:
                print(f"name of {l} or {r} is wrong")
                return False
            if l in leftseated:
                if dml is None:
                    dml = l
                elif l != dml:
                    print(
                        f"we cannot have more than one double match: {l} {dml}")
                    return False
            else:
                leftseated.add(l)

            if r in rightseated:
                print("no double matches for rights possible")
                return False
            else:
                rightseated.add(r)

        return True

    def guess_possible(self, guess: Set[Tuple[str, str]], end) -> bool:
        assert len(guess) <= 11, "The guess cannot have more than 11 pairs"

        if not self.guess_correct_format(guess):
            print("Solution not correct format")
            # print(guess)
            return False

        complete: bool = len(guess) == 11

        nights = self.nights_up_to_episode(end)
        kpm = self.knownpm_up_to_episode(end)

        # no known no matches
        if any([self.no_match(*p, end) for p in guess]):
            ba: List = [self.no_match(*p, end) for p in guess]
            trueindex = ba.index(True)

            # print(f"guess has known no match: {list(guess)[trueindex]} {self.x}")
            return False

        # perfect matches of dmtuple must be the same person
        elif self.dmtuple is not None:
            dml = [l for (l, r) in guess if r in self.dmtuple]
            if len(dml) > 1 and dml[0] != dml[1]:
                print(f"dmtuple does not have same pm {len(guess)}")
                print(guess)
                return False

        if complete:
            # if 11 matches, we must have all perfect matches
            if not all([pm in guess for pm in kpm]):
                print("complete and not all perfect matches")
                return False

        # number of lights with pairs matching lights in nights
        # if we don't have 11 pairs, we allow lesser lights
        for pairs, lights in nights:
            intersection = set(pairs) & guess
            clights = len(intersection)

            if clights > lights:
                return False
            elif clights < lights:
                if complete:
                    return False

        return True

    def generate_complete_guesses(self, end: int, asm: Set[Tuple[str, str]] = set()) -> List[Set[Tuple[str, str]]]:
        '''Generating solutions that fit matchboxes, blackout nights, possible double matches and assumption'''
        if len(asm) == 11:
            return [asm]

        def zip_product(ordering):
            return list(zip(self.lefts, ordering))

        possible_matches = self.get_possible_matches(end)

        if self.dm is not None:
            # lefts who can have a double match
            dm_lefts = [l for l in self.lefts
                        if not self.no_match(l, self.dm, end)]

            asm_dict = {}
            if len(asm) > 0:
                _, a_rights = zip(*asm)
                a_rights = list(set(a_rights) - set([self.dm]))
            else:
                a_rights = []

            for l, r in asm:
                if l not in asm and r != self.dm:
                    asm_dict[l] = [r]

            # remove dm right from possible matches to avoid duplicates solution generation
            modified_pos_matches = {}
            for l in self.lefts:
                if l in asm_dict:
                    modified_pos_matches[l] = asm_dict[l]
                else:
                    modified_pos_matches[l] = [r for r in self.rights
                                               if r not in a_rights and
                                               r != self.dm and
                                               not self.no_match(l, r, end)]

            # get first ten matches
            products = [list(ps) for ps in itertools.product(*modified_pos_matches.values())
                        if len(set(ps)) == 10]
            solutions_tenmatches = list(map(zip_product, products))

            solutions = []
            if self.dmtuple is None:
                pos_additional_matches = [(l, self.dm) for l in dm_lefts]

                for tenmatches, additionalmatch in itertools.product(solutions_tenmatches, pos_additional_matches):
                    if additionalmatch in tenmatches:
                        continue

                    solutions.append(set(tenmatches + [additionalmatch]))

            else:
                # In VIP 2023, we know the pair with the same match
                dm1, dm2 = self.dmtuple
                if dm1 == self.dm:
                    dm1, dm2 = dm2, dm1

                for tenmatches in solutions_tenmatches:
                    dml = [l for (l, r) in tenmatches if r == dm1]
                    sol = set(tenmatches+[(dml[0], dm2)])
                    solutions.append(sol)
        else:
            # Normalo 2023
            asm_dict = {l: [r] for l, r in asm}
            if len(asm) > 0:
                _, a_rights = zip(*asm)
            else:
                a_rights = []

            modified_pos_matches = possible_matches
            modified_pos_matches = {}
            for l in self.lefts:
                if l in asm_dict:
                    modified_pos_matches[l] = asm_dict[l]
                else:
                    modified_pos_matches[l] = [r for r in self.rights
                                               if r not in a_rights and
                                               not self.no_match(l, r, end)]

            # get first ten matches
            products = [list(ps) for ps in itertools.product(*modified_pos_matches.values())
                        if len(set(ps)) == 10]
            solutions_tenmatches = list(map(zip_product, products))

            solutions = []

            for tenmatches in solutions_tenmatches:
                _, rs = zip(*tenmatches)
                msr = list(set(self.rights) - set(rs))[0]
                addmatches = [(l, msr)
                              for l in self.lefts if not self.no_match(l, msr, end)]
                for additionalmatch in addmatches:
                    sol = set(tenmatches + [additionalmatch])
                    if sol not in solutions:
                        solutions.append(sol)
                    if not self.guess_correct_format(sol):
                        print("not self.guess_correct_format(sol)")

        return solutions

    def find_solutions(self, end: int) -> List[Set[Tuple[str, str]]]:
        solutions = self.generate_complete_guesses(end)
        print(len(solutions))

        solutions = list(
            filter(lambda s: self.guess_possible(s, end), solutions))

        return solutions

    def find_solutions_fast(self, end: int) -> List[Set[Tuple[str, str]]]:
        # _, nights, kpm, _, _ = self.get_upto_episode(end)
        nights = self.nights_up_to_episode(end)
        kpm = self.knownpm_up_to_episode(end)

        asm_per_night = []
        for i, (pairs, lights) in enumerate(nights):
            notcorrect = list(filter(lambda p: self.no_match(*p, end),
                                     pairs))
            defcorrect = list(filter(lambda p: p in kpm, pairs))
            remaining = set(pairs) - set(defcorrect) - set(notcorrect)

            combs = [set(comb).union(set(defcorrect))
                     for comb in itertools.combinations(remaining, lights - len(defcorrect))]

            asm_per_night.append(combs)
            # print(i, len(combs))

        merged_asm = functools.reduce(lambda a1, a2: self.merge_guesses_lists(a1, a2),
                                      asm_per_night)
        print("before", len(merged_asm))
        merged_asm = list(filter(lambda a: self.guess_possible(a, end),
                                 merged_asm))
        asm_lengths = list(map(len,merged_asm))
        asm_lengths_counter = Counter(asm_lengths)
        print(asm_lengths_counter)
        exit()

        solutions = []

        for a in merged_asm:
            sols_a = self.generate_complete_guesses(end, a)
            # avoiding duplicate solution
            for s in sols_a:
                if s not in solutions:
                    solutions.append(s)
        print("len(solutions)", len(solutions))

        solutions = list(filter(lambda s: self.guess_possible(s, end),
                                solutions))

        return solutions

    def mergeable_guesses_loop(self, a1: Set, a2: Set) -> bool:
        '''misses some cases'''
        dmc = False
        dml = None

        for (l1, r1), (l2, r2) in itertools.product(a1, a2):
            # two distinct lefts have same match
            if l1 != l2 and r1 == r2:
                return False

            # possible double match
            elif l1 == l2 and r1 != r2:
                if r1 == self.dm or r2 == self.dm:
                    continue

                # check two special cases
                if self.dmtuple is not None:
                    # VIP 2023
                    if r1 not in self.dmtuple or r2 not in self.dmtuple:
                        # print(f"{r1} not in self.dmtuple or {r2} not in self.dmtuple")
                        return False
                elif self.dm == None:
                    # Normalo 2023
                    # Make sure we only have only double match in assumption
                    if dmc or (dml is not None and dml != l1):
                        return False
                    dmc = True
                    dml = l1
                else:
                    return False

        g_union = a1.union(a2)
        if len({l for l, r in g_union if self.dmtuple is not None and r in self.dmtuple}) > 1:
            return False
        elif len(g_union) > 11:
            return False

        return True

    def mergeable_guesses(self, a1: Set, a2: Set) -> bool:
        g_union = a1.union(a2)

        # condition 1: must be no more than 11 pairs
        c1 = len(g_union) <= 11

        sl, sr = zip(*g_union)  # seated lefts and rights
        sl, sr = set(sl), set(sr)
        pd = {l: [r1 for (l1, r1) in g_union if l == l1]
              for l in sl}  # dict for partners
        dls = [l for l in pd if len(pd[l]) == 2]  # lefts appearing twice

        # no double seated rights
        c2 = len(sr) == len(g_union) and \
            len(g_union) - 1 <= len(sl) <= len(g_union)
        # at most one double match
        c3 = len(dls) <= 1

        c4 = True
        if len(dls) > 0:
            dl = dls[0]
            g_dmtuple = pd[dl]
            c4 = (self.dmtuple is None and (self.dm in g_dmtuple)) \
                or (self.dmtuple is not None and set(g_dmtuple) == set(self.dmtuple))

        c5 = True
        if self.dmtuple is not None:
            c5 = len({l for l, r in g_union if r in self.dmtuple}) <= 1

        return all([c1, c2, c3, c4, c5])

    def merge_guesses_lists(self, asm_1: List[Set], asm_2: List[Set]):
        '''
        Input: season, two list of assumptions (set of pairs)
        Output: List of merged together assumptions 
        '''

        m_asm = []
        for a1, a2 in itertools.product(asm_1, asm_2):
            if self.mergeable_guesses_loop(a1, a2) and a1.union(a2) not in m_asm:
                m_asm.append(a1.union(a2))

        return m_asm


def analysize_solutions(season: AYTO, fn: str, end: int) -> None:
    with open(fn, "r") as f:
        content = f.readlines()

    print(f"i: {end} \n")

    mbs = season.matchboxes_up_to_episode(end)
    # print(season.matchboxes_up_to_episode(i+1))

    sols = [eval(line) for line in content]
    nsols = list(filter(lambda sn: season.guess_possible(sn, end), sols))

    print(f"after half: {len(sols)}, after night {end}: {len(nsols)} \n")
    sols = nsols

    pdict = {}
    for s in sols:
        for p in s:
            if p in pdict:
                pdict[p] += 1
            else:
                pdict[p] = 1
    pdict = {p: v for (p, v) in sorted(
        pdict.items(), key=lambda x: x[1], reverse=True)}

    def givescore(s1):
        return sum([pdict[p] for p in s1])

    elevenmostlikely = list(pdict.items())[:11]

    print(f"{elevenmostlikely}\n")

    allpairs = [(l, r) for l in season.lefts for r in season.rights]
    impairs = [p for p in allpairs if p not in pdict and p not in mbs]
    pms = [p for p in allpairs if p in pdict and pdict[p] == len(sols)]
    newpms = [p for p in allpairs if p in pdict and pdict[p]
              == len(sols) and p not in mbs]
    print(f"New pms: {len(newpms)} ({len(pms)}) {newpms}")
    print(f"{len(impairs)}\n")

    # p = ("Leon", "Estelle")
    # if season.no_match(*p, i):
    #     print(f"{p} is definitely no match\n")
    # else:
    #     if p in pdict:
    #         pdict_keys = list(pdict.keys())
    #         print(
    #             f"{p}: {pdict[p]} poss., place {pdict_keys.index(p)} from {len(pdict_keys)}\n")
    #     else:
    #         print(f"{p} is not a match according to left possible solutions\n")

    return


if __name__ == "__main__":
    allseasons = ["normalo2020", "normalo2021", "normalo2022", "normalo2023",
                  "vip2021", "vip2022", "vip2023",
                  "normalo2024"]

    for seasonfn in allseasons[-1:]:
        # with open(f"data/{seasonfn}.json", "r") as f:
        #     seasondata = json.loads(f.read())

        print(seasonfn)
        season = AYTO(seasonfn, nightsfromexcel=True)
        print(season.nights)

        i = 2
        sols1 = season.find_solutions_fast(i)
        print(f"season.find_solutions_fast({i})", len(sols1))
        # print()

        # analysize_solutions(season, f"analytics/{seasonfn}_half.txt", i)
