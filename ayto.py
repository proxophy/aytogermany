from collections import Counter
import itertools
import functools
import time
import pandas as pd


def time_it(inner):
    @functools.wraps(inner)
    def c_inner(*args):
        start = time.time()
        res = inner(*args)
        end = time.time()
        print(f"=== time needed for {inner.__name__}: {(end-start):0.3f}s ===")
        return res
    return c_inner


PartialSol = set[tuple[str, str]]
CompleteSol = set[tuple[str, str]]
Night = tuple[list[tuple[str, str]], int]
Matchboxes = dict[tuple[str, str], bool]


class AYTO:

    lefts: list[str]  # gender group of 10
    rights: list[str]  # gender group of 11

    nights: list[Night]
    matchboxes: Matchboxes
    bonights: list[int]

    dm: str | None
    dmtuple: tuple[str, str] | None

    numepisodes: int
    knownnights: list[int]
    knownboxes: list[int]
    knownpms: list[tuple[str, str]]

    solution: CompleteSol | None

    def __init__(self,
                 lefts: list[str], rights: list[str],
                 nights: list[Night],
                 matchboxes: Matchboxes = {},
                 bonights: list[int] = [],
                 dm: str | None = None, dmtuple: tuple[str, str] | None = None,
                 cancellednight: int = -1, boxesepisodes: list[int] = list(range(10)),
                 solution: CompleteSol | None = None) -> None:

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
        self.dmtupleknown = 7
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

    def get_nights(self, options: dict) -> list[Night]:
        """Retrieve nights based on episode limits."""
        end = min(self.numepisodes - 1, options.get("end", self.numepisodes - 1))
        nights = self.nights[: self.knownnights[end] + 1]
        return nights if options.get("includenight", True) else nights[:-1]

    def get_matchboxes(self, options: dict) -> Matchboxes:
        end = min(self.numepisodes-1,
                  options.get("end", self.numepisodes - 1))
        usedmbkeys = list(self.matchboxes.keys())[:(self.knownboxes[end]+1)]
        usedmb = {k: self.matchboxes[k] for k in usedmbkeys}
        return usedmb

    def get_bonights(self, options: dict) -> list[int]:
        end = min(self.numepisodes-1,
                  options.get("end", self.numepisodes - 1))
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
        if not mb.get((l, r), True):
            return True

        # one is part of known perfect match
        # consider multiple seated lefts
        if (l, r) not in kpm and ((l in hpm) or (r in hpm)) and r != self.dm:
            return True

        return False

    def parsol_correct_format(self, parsol: PartialSol) -> bool:
        if len(parsol) > self.nummatches:
            print("len(solution) > self.nummatches")
            return False

        leftseated = set()
        rightseated = set()
        dml = None

        for l, r in parsol:
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

    def parsol_possible(self, parsol: PartialSol,  options: dict) -> bool:
        if len(parsol) > self.nummatches:
            return False
        # assert len(parsol) <= self.nummatches, \
        #     f"The partial solution has {len(parsol)} instead of {self.nummatches}"
        checknights: bool = options.get("checknights", True)
        end: int = options.get("end", self.numepisodes-1)

        if not self.parsol_correct_format(parsol):
            return False

        complete: bool = len(parsol) == self.nummatches

        kpm = self.get_pms(options)

        # no known no matches
        if any([self.no_match(*p,  options) for p in parsol]):
            # ba: list = [self.no_match(*p,  options) for p in parsol]
            # trueindex = ba.index(True)
            # print(f"parsol has known no match: {list(parsol)[trueindex]}")
            return False

        # VIP 23
        # perfect matches of dmtuple must be the same person
        elif (self.dmtuple is not None and end >= self.dmtupleknown):
            dml = [l for (l, r) in parsol if r in self.dmtuple]
            if len(dml) > 1 and dml[0] != dml[1]:
                # print(f"dmtuple rights do not have same pm {len(parsol)}")
                return False

        # check condition for double matches
        pdict = {l: [] for l in self.lefts}
        for (l, r) in parsol:
            pdict[l].append(r)
        g_lefts, g_rights = zip(*parsol)

        # lefts with multiple matches
        mutiplels = [l for l in self.lefts if len(pdict[l]) > 1]
        if len(mutiplels) > 1:
            print("Only one double/tripple match")
            return False
        elif len(mutiplels) == 1:
            mutiplel = mutiplels[0]
            mutiplers = pdict[mutiplel]

            if (self.dmtuple is not None and end >= self.dmtupleknown) \
                    and len(mutiplers) == 2 and set(self.dmtuple) != set(mutiplers):
                return False
            elif self.dm is not None and len(mutiplers) == 2 and self.dm not in pdict[mutiplel]:
                return False

        # if parsol has nummatches matches
        if complete:
            # we must have all perfect matches
            if not all([pm in parsol for pm in kpm]):
                print("complete and not all perfect matches in partial sol")
                return False

            # we must have 10 seated lefts and nummatches rights
            g_lefts, g_rights = zip(*parsol)
            if len(set(g_lefts)) != 10 or len(set(g_rights)) != self.nummatches:
                return False

        if not checknights:
            return True

        # number of lights with pairs matching lights in nights
        # if we don't have self.nummatches pairs, we allow lesser lights
        nights = self.get_nights(options)
        for pairs, lights in nights:
            intersection = set(pairs) & parsol
            clights = len(intersection)

            if clights > lights:
                return False
            elif clights < lights:
                if complete:
                    return False

        return True

    def merge_parsols_lists(self, psl_1: list[PartialSol], psl_2: list[PartialSol],
                            options: dict):
        '''
        Input: season, two list of partial sols (set of pairs)
        Output: list of merged together partial sols 
        '''
        m_asm = []
        for g1, g2 in itertools.product(psl_1, psl_2):
            pred = self.parsol_possible(
                g1.union(g2), options)
            if pred and g1.union(g2) not in m_asm:
                m_asm.append(g1.union(g2))

        return m_asm

    def get_parsol_leftrights(self, parsol: PartialSol):
        if len(parsol) > 0:
            g_lefts, g_rights = zip(*parsol)
            g_lefts, g_rights = list(g_lefts), list(g_rights)
        else:
            g_lefts, g_rights = set(), set()
        return set(g_lefts), set(g_rights), len(g_lefts) - len(set(g_lefts)) > 0

    def possible_matches_for_parsol(self, parsol: PartialSol, options: dict) -> dict[str, list[str]]:
        g_lefts, g_rights, _ = self.get_parsol_leftrights(parsol)

        nights = self.get_nights(options)
        sitting_nomatches = {}
        # Consider sitting matches as no matches if not in partial sol
        for pairs, _ in nights:
            for p in set(pairs) - parsol:
                sitting_nomatches[p] = True

        pos_matches = {l: [r for r in set(self.rights) - g_rights
                           if not self.no_match(l, r,  options)
                           and not sitting_nomatches.get((l, r), False)
                           and r != self.dm]
                       for l in self.lefts if l not in g_lefts}
        return pos_matches

    def merge_mm_in_parsol(self, psol: PartialSol, others_list: list[PartialSol], options: dict):
        _, _, dm_in_parsol = self.get_parsol_leftrights(psol)

        if dm_in_parsol:
            return [psol.union(set(othermatches)) for othermatches in others_list]
        return []

    def merge_mm_not_in_parsol(self, psol: PartialSol, others_list: list[PartialSol], options: dict):
        end: int = options.get("end", self.numepisodes-1)

        nights = self.get_nights(options)
        sitting_nomatches = {}
        # Consider sitting matches as no matches if not in parsol
        for pairs, _ in nights:
            for p in set(pairs) - psol:
                sitting_nomatches[p] = True

        solutions = []
        addmatches_dict = {r: [(l, r) for l in self.lefts
                               if not (self.no_match(l, r, options) or sitting_nomatches.get((l, r), False))]
                           for r in self.rights}

        for othermatches in others_list:
            tenmatches = psol.union(othermatches)
            assert len(tenmatches) == 10

            _, crights = zip(*tenmatches)
            missingright = [r for r in self.rights if r not in crights][0]

            # VIP 2023: dmtuple
            if self.dmtuple is not None and end >= self.dmtupleknown:
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
                    solutions += [tenmatches.union([ap])
                                  for ap in addmatches_dict[missingright]]
                else:
                    dmleft = [
                        l for (l, r) in tenmatches if r == self.dm][0]
                    if (dmleft, missingright) not in addmatches_dict[missingright]:
                        continue
                    solutions.append(tenmatches.union(
                        [(dmleft, missingright)]))
        return solutions

    def generate_complete_solutions(self, psol: PartialSol, options: dict, flag: bool) -> list[CompleteSol]:
        '''Generating solutions'''
        if len(psol) == self.nummatches:
            return [psol]

        def zip_product(clefts, ordering):
            return set(zip(clefts, ordering))

        _, _, dm_in_parsol = self.get_parsol_leftrights(psol)
        pos_matches = self.possible_matches_for_parsol(psol, options)

        # for l in pos_matches:
        #     print(l, pos_matches[l])

        products = [list(ps) for ps in itertools.product(*pos_matches.values())
                    if len(set(ps)) == len(ps)]
        others_list = list(map(lambda p: zip_product(pos_matches.keys(), p),
                               products))

        if dm_in_parsol > 0:
            # DM is already in parsol
            return self.merge_mm_in_parsol(psol, others_list, options)

        solutions = self.merge_mm_not_in_parsol(
            psol, others_list, options)

        unique_sols = []

        for sol in solutions:
            if len(sol) != self.nummatches:
                raise ValueError
            _, rs = zip(*sol)
            if 2 in Counter(rs).values():
                raise ValueError

            if sol not in unique_sols:
                unique_sols.append(sol)

        return unique_sols

    def generate_parsols(self, options: dict) -> list[PartialSol]:
        verbose: bool = options["verbose"]

        if verbose:
            print("generate_parsols")

        nights = self.get_nights(options)
        kpm = self.get_pms(options)

        parsols_per_night = []
        for pairs, lights in nights:
            notcorrect = list(filter(lambda p: self.no_match(*p, options) or p in kpm,
                                     pairs)) 
            defcorrect = list(filter(lambda p: p in kpm, pairs))
            remaining = set(pairs) - set(defcorrect) - set(notcorrect)

            combs = [set(comb).union(kpm)
                     for comb in itertools.combinations(remaining, lights - len(defcorrect))]

            parsols_per_night.append(combs)

        merged_parsols: list[PartialSol] = functools.reduce(lambda g1, g2: self.merge_parsols_lists(g1, g2,  options),
                                                            parsols_per_night)

        merged_parsols: list[PartialSol] = list(filter(lambda a: self.parsol_possible(a, options),
                                                       merged_parsols))

        if verbose:
            parsols_lengths_counter = Counter(list(map(len, merged_parsols)))
            print(f"Parsols lengths: {parsols_lengths_counter}")

        return merged_parsols


def find_solutions_slow(season: AYTO, options: dict) -> list[CompleteSol]:
    solutions = season.generate_complete_solutions(set(),  options, False)
    print(f"Generated solutions: {len(solutions)}")
    solutions = list(
        filter(lambda s: season.parsol_possible(s, options), solutions))

    return solutions


def find_solutions(season: AYTO, options: dict, flag: bool, asm: list = []) -> list[CompleteSol]:
    start = time.time()
    times = []
    verbose: bool = options.get("verbose", False)

    merged_parsols = season.generate_parsols(options)

    times.append(time.time()-start)
    start = time.time()

    solutions_unfiltered: list[CompleteSol] = []
    for g in merged_parsols:
        sols_g = season.generate_complete_solutions(g,  options, flag)
        solutions_unfiltered += sols_g

    times.append(time.time()-start)
    start = time.time()

    # options.update({"checknights": True})
    solutions = list(filter(lambda s: season.parsol_possible(s, options),
                            solutions_unfiltered))
    if len(solutions_unfiltered) - len(solutions) > 0:
        print("len(solutions_unfiltered) - len(solutions) > 0")

    times.append(time.time()-start)
    if verbose:
        print(f"Generating parsols: {times[0]:0.2f} s")
        print(f"Generating solutions: {times[1]:0.2f} s")
        print(f"Filtering solutions: {times[2]:0.2f} s")
        print(
            f"Before and after filtering: {len(solutions_unfiltered)} {len(solutions)}")

    if len(asm) > 0:
        # keep solutions that have at least one pair of assumption
        solutions_unfiltered = [sol for sol in solutions_unfiltered
                                if any([p in sol for p in asm])]

    return solutions


def dm_left(sol: CompleteSol) -> str:
    '''Which of lefts has double match'''
    lefts = list(zip(*sol))[0]
    return [l for (l, r) in Counter(lefts).items() if r == 2][0]


@time_it
def analysize_solutions(season: AYTO, options: dict, asm: list = []):
    mbs = season.get_matchboxes(options)
    sols = find_solutions(season,  options, False, asm)
    end = options.get("end", season.numepisodes-1)

    pairs_counter = Counter([p for s in sols for p in s])

    allpairs = itertools.product(season.lefts, season.rights)
    perfect_matches = [
        p for p in pairs_counter if pairs_counter[p] == len(sols)]
    new_nomatches = list(filter(lambda p: not (season.no_match(*p, options) or p in pairs_counter),
                                allpairs))
    new_pms = list(filter(lambda p: p not in mbs, perfect_matches))

    print(f"Nach {end+1} Doppelfolgen")
    print(f"Anzahl Möglichkeiten: {len(sols)}")
    print(f"Bekannte Perfect Matches: {perfect_matches}")
    if len(new_pms) > 0:
        print(f"Neue Perfect Matches: {new_pms}")
    if len(new_nomatches) > 0:
        impdict = {l: [] for l in season.lefts}
        for l, r in new_nomatches:
            impdict[l].append(r)
        print("Neue No-Matches durch Ausschlussprinzip:")
        for l in impdict:
            if len(impdict[l]) == 0:
                continue
            print(f"{l}: {', '.join(impdict[l])}")
    # Wie wahrscheinlich hat ein left ein Doppelmatch
    dm_lefts = [(l, round(r/len(sols)*100, 1))
                for (l, r) in Counter([dm_left(s) for s in sols]).items()]
    dm_lefts.sort(key=(lambda a: a[1]), reverse=True)
    print(f"Person mit Doppelmatch: {dm_lefts}")

    data = {l: pd.Series([round(pairs_counter.get((l, r), 0)/len(sols)*100, 1) for r in season.rights],
                         index=season.rights) for l in season.lefts}

    df = pd.DataFrame(data)

    return df


def matching_night_probs(season: AYTO, episode: int):
    options = {"end": episode,
               "includenight": False,
               "verbose": False}
    beforenight = find_solutions(season, options, False)
    night = season.get_nights({"end": episode,
                               "includenight": True})[-1][0]
    nightpossol = any([set(night).issubset(sol) for sol in beforenight])

    print(f"Pairs of nights are possible solution: {nightpossol}")

    poslights = Counter([len(set(night).intersection(sol))
                        for sol in beforenight])
    return [round(poslights.get(i, 0)/len(beforenight)*100, 2) for i in range(0, 11)]
