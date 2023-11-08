from typing import List, Tuple, Dict, Set, Union
import itertools
import json
import functools


class AYTO():
    data: Dict

    lefts: List[str]
    rights: List[str]

    nights: List[Tuple[List[Tuple[str, str]], int]]
    matchboxes: Dict[Tuple[str, str], bool]
    bonights: List[int]

    dm: Union[str, None]
    dmtuple: Union[Tuple[str, str], None]

    cancellednight: int
    numepisodes: int
    knownnights: List[int]
    knownboxes: List[int]
    knownpm: List[Tuple[str, str]]
    knowhaspm: List[str]

    solution: Union[None, List[Tuple[str, str]]]

    def __init__(self, data) -> None:
        self.data = data

        # read data from provided json data structure

        # set left and right
        if len(data["women"]) == 11 and len(data["men"]) == 10:
            self.lefts, self.rights = data["men"], data["women"]
        elif len(data["women"]) == 10 and len(data["men"]) == 11:
            self.lefts, self.rights = data["women"], data["men"]
        else:
            raise ValueError(f"not enough or too much women or men")

        # seatings and lights for nights
        self.read_nights()

        # matchboxes with results
        self.read_matchboxes()

        # nights with blackout
        self.bonights = data["bonights"]

        # do we know who is the second match to someone
        # besides normalo2023, yes
        if "dm" in data:
            self.dm = data["dm"]
        else:
            self.dm = None
            # self.dm_rights = [r for r in self.rights if r not in self.haspm]

        # do we know which two share the same match
        # besides vip2023, no
        if "dmtuple" in data:
            dm1, dm2 = data["dmtuple"]
            if not (dm1 in self.rights and dm2 in self.rights):
                raise ValueError(
                    f"{dm1} or {dm2} may be misspelled for dmtuple")
            self.dmtuple = (dm1, dm2)
        else:
            self.dmtuple = None

        # added to do some funny analyzing
        if "cancelled" in data:
            self.cancellednight = data["cancelled"]
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

        self.read_boxesepisode()

        # known perfect matches
        self.knownpm = [p for p in self.matchboxes.keys()
                        if self.matchboxes[p]]
        # people who have known perfect match
        self.haspm = [e for p in self.knownpm for e in p]

        if "solution" in data:
            self.solution = data["solution"]
            print(self.solution)

    def read_nights(self) -> None:
        '''Read nights and check if everything is fine'''
        # assert that nights are in data
        assert "nights" in self.data, "data has not a night key"

        self.nights = []

        # make sure all pairs are valid pairs and there are no duplicates
        for i, (pairs, lights) in enumerate(self.data["nights"]):
            self.nights.append(([], lights))

            lefts, rights = [], []

            for l, r in pairs:
                if l in self.lefts and r in self.rights:
                    # check that we don't have double seatings
                    if l in lefts or r in rights:
                        raise ValueError(f"{l} or {r} has already been seated")
                    lefts.append(l)
                    rights.append(r)

                    self.nights[i][0].append((l, r))
                elif l in self.rights and r in self.lefts:
                    # check that we don't have double seatings
                    if l in rights or r in lefts:
                        raise ValueError(f"{l} or {r} has already been seated")
                    lefts.append(r)
                    rights.append(l)

                    self.nights[i][0].append((r, l))
                else:
                    raise ValueError(
                        f"{l} or {r} is neither in the list of women or men")

        return

    def read_matchboxes(self) -> None:
        '''Read matchboxes'''
        assert "matchboxes" in self.data, "data has not a night key"

        self.matchboxes = {}

        for (l, r), result in self.data["matchboxes"]:
            if l in self.lefts and r in self.rights:
                if (l, r) in self.matchboxes:
                    raise ValueError(f"Double matchbox result for {l,r}")
                self.matchboxes[(l, r)] = result
            elif l in self.rights and r in self.lefts:
                if (r, l) in self.matchboxes:
                    raise ValueError(f"Double matchbox result for {r,l}")
                self.matchboxes[(r, l)] = result

    def read_boxesepisode(self) -> None:
        if "boxesepisode" in self.data:
            assert len(self.matchboxes) == len(self.data["boxesepisode"]), \
                r"not every matchbox could be assigned a episode"

            self.knownboxes = [0 for _ in range(self.numepisodes)]
            for i, e in enumerate(self.data["boxesepisode"]):
                self.knownboxes[e] = i
            for i in range(1, len(self.knownboxes)):
                if self.knownboxes[i] == 0:
                    self.knownboxes[i] = self.knownboxes[i - 1]
        else:
            self.knownboxes = [i for i in range(len(self.nights))]

    def solution_correct_format(self, solution: Set[Tuple[str, str]]) -> bool:
        if len(solution) != 11:
            # print(solution)
            print("len(solution) != 11")
            return False

        leftseated = {l: 0 for l in self.lefts}
        rightseated = {r: 0 for r in self.rights}

        for l, r in solution:
            if l not in self.lefts or r not in self.rights:
                print(f"{l} {r}")
                return False
            leftseated[l] += 1
            rightseated[r] += 1

        if 0 in leftseated.values() or 0 in rightseated.values():
            print("0 in leftseated.values() or 0 in rightseated.values()")
            print(solution)
            print(leftseated)
            print(rightseated)
            return False
        return True

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

    def guess_possible(self, guess: Set[Tuple[str, str]], end) -> bool:
        assert len(guess) <= 11, "The guess cannot have more than 11 pairs"

        complete: bool = len(guess) == 11

        nights = self.nights_up_to_episode(end)
        kpm = self.knownpm_up_to_episode(end)

        # no known no matches
        if any([self.no_match(*p, end) for p in guess]):
            print("guess has known no match")
            return False

        # perfect matches of dmtuple must be the same person
        elif self.dmtuple is not None:
            print("dmtuple does not have same pm")
            dml = [l for (l, r) in guess if r in self.dmtuple]
            if len(dml) > 1 and dml[0] != dml[1]:
                return False

        if complete:
            # complete solution must be of correct format
            if not self.solution_correct_format(guess):
                print("Solution not correct format")
                return False

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

    def generate_solutions(self, end: int, asm: Set[Tuple[str, str]] = set()) -> List[Set[Tuple[str, str]]]:
        '''Generating solutions that fit matchboxes, blackout nights, possible double matches and assumption'''
        def zip_product(ordering):
            return list(zip(self.lefts, ordering))

        possible_matches = self.get_possible_matches(end)

        if self.dm is not None:
            # lefts who can have a double match
            dm_lefts = [
                l for l in self.lefts if not self.no_match(l, self.dm, end)]

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
                    if not self.solution_correct_format(sol):
                        print("not self.solution_correct_format(sol)")

        return solutions

    def find_solutions(self, end: int) -> List[Set[Tuple[str, str]]]:
        solutions = self.generate_solutions(end)
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
            print(i, len(combs))

        merged_asm = functools.reduce(lambda a1, a2: self.merge_assumptions_lists(a1, a2),
                                      asm_per_night)
        print("before", len(merged_asm))
        merged_asm = list(filter(lambda a: self.guess_possible(a, end),
                                 merged_asm))
        print("after", len(merged_asm))
        solutions = []

        for a in merged_asm:
            solutions += self.generate_solutions(end, a)
        print(len(solutions))

        solutions = list(filter(lambda s: self.guess_possible(s, end),
                                solutions))

        return solutions

    def merge_assumptions_lists(self, asm_1: List[Set], asm_2: List[Set]):
        '''
        Input: season, two list of assumptions (set of pairs)
        Output: List of merged together assumptions 
        '''

        m_asm = []
        for a1, a2 in itertools.product(asm_1, asm_2):
            possible = True
            dmc = 0
            for (l1, r1), (l2, r2) in itertools.product(a1, a2):
                if l1 != l2 and r1 == r2:
                    possible = False
                # double match
                elif l1 == l2 and r1 != r2:
                    # print(f"{l1,r1}, {l2,r2}")
                    if r1 == season.dm or r2 == season.dm:
                        continue
                    elif season.dm == None:
                        # for Normalo 2023: make sure we only have only double match in assumption
                        dmc += 1
                        if dmc > 1:
                            possible = False
                    else:
                        possible = False
                    # TODO: adjust for unknown dm and dm couple
                if not possible:
                    break

            if len(a1.union(a2)) > 11:
                possible = False

            if possible and a1.union(a2) not in m_asm:
                m_asm.append(a1.union(a2))

        return m_asm


def analysize_solutions(season, fn: str, end: int) -> None:
    with open(fn, "r") as f:
        currentsols = eval(f.readlines()[-1])

    possols = list(
        filter(lambda s: season.guess_solution(s, end), currentsols))

    print(f"len(sols): {len(currentsols)}, len(possols): {len(possols)}")
    if len(possols) < len(currentsols):
        print("writing new solutions in file")
        with open(fn, "a") as f:
            f.write(f"\n \n{str(possols)}")

    # returning possible matches
    pms = {}
    for sol in possols:
        print(sol)
        for l, r in sol:
            if (l, r) in pms:
                pms[l, r] += 1
            else:
                pms[l, r] = 1

    import pandas
    fields = [""] + season.rights
    rows = [[l] + [str(pms[(l, r)]) if (l, r) in pms else "0" for r in season.rights]
            for l in season.lefts]
    df = pandas.DataFrame(data=rows, columns=fields)
    print(df)

    return


if __name__ == "__main__":
    allseasons = ["normalo2020", "normalo2021", "normalo2022", "normalo2023",
                  "vip2021", "vip2022", "vip2023"]

    for seasonfn in allseasons[:1]:
        # if seasonfn == "normalo2022":
        #     # skip this because it's very slow
        #     continue

        with open(f"data/{seasonfn}.json", "r") as f:
            seasondata = json.loads(f.read())

        print(seasonfn)
        season = AYTO(seasondata)

        i = 5
        sols1 = season.find_solutions_fast(i)
        print(f"season.analyze_nights({i})", len(sols1))
        sols2 = season.find_solutions(i)
        print(f"season.find_solutions({i})", len(sols2))
