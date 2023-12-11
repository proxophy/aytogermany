from typing import List, Tuple, Dict, Set, Optional
from collections import Counter
import itertools
import functools
import time
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

    # cancellednight: int
    numepisodes: int
    knownnights: List[int]
    knownboxes: List[int]
    knownpm: List[Tuple[str, str]]

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
            self.numepisodes = len(self.nights)
            self.knownnights = [i for i in range(self.numepisodes)]
        else:
            self.numepisodes = len(self.nights) + 1
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
        self.knownpm = [p for p in self.matchboxes.keys()
                        if self.matchboxes[p]]
        # people who have known perfect match
        self.haspm = [e for p in self.knownpm for e in p]

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

    def get_knowpms(self, end: int) -> List[Tuple[str, str]]:
        mb = self.get_matchboxes(end)
        return list(set(mb.keys()) & set(self.knownpm))

    def no_match(self, l: str, r: str, end: int) -> bool:
        '''(l,r) are definitely no match'''
        assert l in self.lefts and r in self.rights, f"f: {l}, s: {r}"

        nights = self.get_nights(end)
        mb = self.get_matchboxes(end)
        bon = self.get_bonights(end)
        kpm = self.get_knowpms(end)
        hpm = [e for p in kpm for e in p]

        # one is part of known perfect match
        # matchbox result was false
        # pair in blackout night who is not known perfect match
        return ((l, r) not in kpm and ((l in hpm) or (r in hpm))) \
            or ((l, r) in mb.keys() and not mb[(l, r)]) \
            or any([(l, r) in nights[bo][0] and (l, r) not in kpm for bo in bon])

    def guess_correct_format(self, guess: Set[Tuple[str, str]]) -> bool:
        if len(guess) > self.nummatches:
            print("len(solution) > self.nummatches")
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
        assert len(guess) <= self.nummatches, \
            f"The guess has {len(guess)} instead of {self.nummatches}"

        if not self.guess_correct_format(guess):
            print("Solution not correct format")
            # print(guess)
            return False

        complete: bool = len(guess) == self.nummatches

        nights = self.get_nights(end)
        kpm = self.get_knowpms(end)

        # no known no matches
        if any([self.no_match(*p, end) for p in guess]):
            ba: List = [self.no_match(*p, end) for p in guess]
            trueindex = ba.index(True)
            print(f"guess has known no match: {list(guess)[trueindex]}")
            return False

        # perfect matches of dmtuple must be the same person
        elif self.dmtuple is not None:
            dml = [l for (l, r) in guess if r in self.dmtuple]
            if len(dml) > 1 and dml[0] != dml[1]:
                print(f"dmtuple does not have same pm {len(guess)}")
                return False

        if complete:
            # if guess has self.nummatches matches, we must have all perfect matches
            if not all([pm in guess for pm in kpm]):
                print("complete and not all perfect matches")
                return False

        # number of lights with pairs matching lights in nights
        # if we don't have self.nummatches pairs, we allow lesser lights
        for pairs, lights in nights:
            intersection = set(pairs) & guess
            clights = len(intersection)

            if clights > lights:
                return False
            elif clights < lights:
                if complete:
                    return False

        return True

    def mergeable_guesses_loop(self, g1: Set, g2: Set) -> bool:
        '''misses some cases'''
        dmc = False
        dml = None

        for (l1, r1), (l2, r2) in itertools.product(g1, g2):
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

        g_union = g1.union(g2)
        if len({l for l, r in g_union if self.dmtuple is not None and r in self.dmtuple}) > 1:
            return False
        elif len(g_union) > self.nummatches:
            return False

        return True

    def mergeable_guesses(self, g1: Set, g2: Set) -> bool:
        g_union = g1.union(g2)

        # condition 1: must be no more than 11 pairs
        c1 = len(g_union) <= self.nummatches

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

    def generate_solutions(self, end: int, asm: Set[Tuple[str, str]] = set()) -> List[Set[Tuple[str, str]]]:
        '''Generating solutions'''
        if len(asm) == self.nummatches:
            return [asm]

        def zip_product(ordering):
            return list(zip(self.lefts, ordering))

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
            # print("before", len(solutions_tenmatches))
            solutions_tenmatches = list(filter(lambda g: self.guess_possible(set(g), end),
                                               solutions_tenmatches))
            # print("after", len(solutions_tenmatches))

            solutions = []
            raddmatch = {r: [(l, r) for l in self.lefts if not self.no_match(l, r, end)]
                         for r in self.rights}

            for tenmatches in solutions_tenmatches:
                _, rs = zip(*tenmatches)
                msr = list(set(self.rights) - set(rs))[0]
                addmatches = raddmatch[msr]

                for additionalmatch in addmatches:
                    sol = set(tenmatches + [additionalmatch])
                    if sol not in solutions:
                        solutions.append(sol)
                    # if not self.guess_correct_format(sol):
                    #     print("not self.guess_correct_format(sol)")

        solutions2 = []
        for sol in solutions:
            if sol not in solutions2:
                solutions2.append(sol)

        return solutions2

    def generate_solutions2(self, end: int, asm: Set[Tuple[str, str]] = set()) -> List[Set[Tuple[str, str]]]:
        '''Generating solutions'''
        if len(asm) == self.nummatches:
            return [asm]

        def zip_product(clefts, ordering):
            return set(zip(clefts, ordering))

        if len(asm) > 0:
            a_lefts, a_rights = zip(*asm)
            a_lefts, a_rights = list(a_lefts), list(a_rights)
        else:
            a_lefts, a_rights = {}, {}

        # are the double matches in asm
        dmpairs_in_asm = len(set(a_lefts)) != len(a_lefts)

        # is double/tripple match right in asm
        dm_in_asm = self.dm is not None and self.dm in a_rights
        tm_in_asm = self.tm is not None and self.tm in a_rights
        dmt = self.dmtuple is not None
        dmt_in_asm = dmt and (
            self.dmtuple[0] in a_rights or self.dmtuple[1] in a_rights)  # type: ignore

        # possible matches for the lefts not assigned
        pos_matches = {l: [r for r in self.rights
                           if r not in a_rights and not self.no_match(l, r, end) and
                           (tm_in_asm or r != self.tm) and
                           (not dm_in_asm or r != self.dm) and
                           (not dmt_in_asm or (dmt and r not in self.dmtuple)) and # type: ignore
                           (dmt_in_asm or r != self.dm)]  # type: ignore
                       for l in self.lefts if l not in a_lefts}

        for l in pos_matches:
            print(l, pos_matches[l])

        products = [list(ps) for ps in itertools.product(*pos_matches.values())
                    if len(set(ps)) == len(ps)]
        others_list = list(map(lambda p: zip_product(pos_matches.keys(), p),
                               products))
        # print("len(products)", len(products))
        if dmpairs_in_asm:
            # print("DM IN ASM")
            return [asm.union(set(othermatches)) for othermatches in others_list]

        solutions = []

        addmatches_dict = {r: [(l, r) for l in self.lefts if not self.no_match(l, r, end)]
                           for r in self.rights}

        # if self.dm is not None:
        for othermatches in others_list:
            tenmatches = asm.union(othermatches)
            _, crights = zip(*tenmatches)
            missingright = [r for r in self.rights if r not in crights][0]

            if self.tm is not None:
                missingrights = [r for r in self.rights if r not in crights]
                mr1, mr2 = missingrights
                if self.tm in missingrights:
                    for l in self.lefts:
                        addmatches = [(l, mr1), (l, mr2)]
                        if all([not self.no_match(*p, end) for p in addmatches]):
                            solutions.append(tenmatches.union(addmatches))
                else:
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
        duplicates = 0

        for sol in solutions:
            if sol not in solutions2:
                solutions2.append(sol)
            else:
                duplicates += 1

            if len(sol) != self.nummatches:
                raise ValueError
            ls, rs = zip(*sol)
            if 2 in Counter(rs).values():
                raise ValueError
            
        # print(duplicates)
        if duplicates > 0:
            print("duplicates", duplicates)

        return solutions2

    def compute_guesses(self, end: int) -> List[Set[Tuple[str, str]]]:
        nights = self.get_nights(end)
        kpm = self.get_knowpms(end)

        guesses_per_night = []
        for pairs, lights in nights:
            notcorrect = list(filter(lambda p: self.no_match(*p, end),
                                     pairs))
            defcorrect = list(filter(lambda p: p in kpm, pairs))
            remaining = set(pairs) - set(defcorrect) - set(notcorrect)

            combs = [set(comb).union(set(defcorrect))
                     for comb in itertools.combinations(remaining, lights - len(defcorrect))]

            guesses_per_night.append(combs)

        merged_guesses = functools.reduce(lambda g1, a2: merge_guesses_lists(self,g1, a2),
                                          guesses_per_night)

        merged_guesses = list(filter(lambda a: self.guess_possible(a, end),
                                     merged_guesses))

        guesses_lengths_counter = Counter(map(len, merged_guesses))
        print(f"Guesses lengths: {guesses_lengths_counter}")

        return merged_guesses


def find_solutions_slow(season: AYTO, end: int) -> List[Set[Tuple[str, str]]]:
    solutions = season.generate_solutions2(end)
    print(f"Generated solutions: {len(solutions)}")

    solutions = list(
        filter(lambda s: season.guess_possible(s, end), solutions))

    return solutions

def merge_guesses_lists(season, gsl_1: List[Set], gsl_2: List[Set]):
    '''
    Input: season, two list of assumptions (set of pairs)
    Output: List of merged together assumptions 
    '''

    m_asm = []
    for g1, g2 in itertools.product(gsl_1, gsl_2):
        # self.mergeable_guesses_loop misses some cases when DM is not known, but is faster
        if season.dm is not None:
            pred = season.mergeable_guesses_loop(g1, g2)
        else:
            pred = season.mergeable_guesses(g1, g2)
        if pred and g1.union(g2) not in m_asm:
            m_asm.append(g1.union(g2))

    return m_asm

def find_solutions(season : AYTO, end: int = 9, old: bool =True) -> List[Set[Tuple[str, str]]]:
    merged_guesses = season.compute_guesses(end)

    solutions = []
    duplicates = 0
    for a in merged_guesses:
        if old:
            sols_a = season.generate_solutions(end, a)
        else:
            sols_a = season.generate_solutions2(end, a)

        # avoiding duplicate solutions
        for s in sols_a:
            if s not in solutions:
                solutions.append(s)
            else:
                duplicates += 1
    print(
        f"Number of generated solutions: {len(solutions)}; Duplicates: {duplicates}")

    solutions = list(filter(lambda s: season.guess_possible(s, end),
                            solutions))

    return solutions

if __name__ == "__main__":
    allseasons = ["normalo2020", "normalo2021", "normalo2022", "normalo2023",
                  "vip2021", "vip2022", "vip2023",
                  "normalo2024"]

    for seasonfn in allseasons[7:]:
        # with open(f"data/{seasonfn}.json", "r") as f:
        #     seasondata = json.loads(f.read())

        print(seasonfn)
        season: AYTO = utils.read_data(seasonfn)

        i = 4
        guesses = season.compute_guesses(i)

        start = time.time()
        
        # for g in guesses:
        #     if len(g) == 8:
        #        print(g)
        # print(time.time()-start)


        # with Mela
        gr1 = {('Martin', 'Lina'), ('Kevin', 'Mela'), ('Paddy', 'Maja'), ('Gerrit', 'Pia'),
               ('Ryan', 'Jana'),  ('Sandro', 'Sina'), ('Sidar', 'Afra'), ('Paulo', 'Lisa-Marie')}
        # without Mela, old: 54/66, new: 54/60
        gr2 = {('Paddy', 'Shelly'), ('Gerrit', 'Pia'), ('Sandro', 'Sina'), ('Eti', 'Maja'),
               ('Ryan', 'Jana'), ('Sidar', 'Afra'), ('Paulo', 'Lisa-Marie'), ('Martin', 'Julia')}

        gr = gr2

        gsols = season.generate_solutions2(i, gr)
        print("len(gsols)", len(gsols))

        # sols1 = find_solutions_slow(season,i)
        # sols1 = find_solutions(season,i, True)
        # sols2 = find_solutions(season,i, False)
        # print("len(sols)", len(sols1), len(sols2))
        
        # print(len(sols2))
        # with open("normalo20242.txt","w") as f:
        #     f.write(str(sols1))

