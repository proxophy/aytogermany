from typing import List, Tuple, Dict, Set, Union
import itertools
from copy import deepcopy
import json


class AYTO():
    lefts: List[str]
    rights: List[str]
    data: Dict
    nights: List[Tuple[List[Tuple[str, str]], int]]
    matchboxes: Dict[Tuple[str, str], bool]
    possible_matches: Dict[str, List[str]]
    dmtuple: Union[Tuple[str, str], None]

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

    def read_nights(self) -> None:
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

    def get_upto_episode(self, end):
        if end >= self.numepisodes - 1:
            return self.matchboxes, self.nights, self.knownpm, self.haspm, self.bonights

        # only consider data upto and including night end
        usedmbkeys = list(self.matchboxes.keys())[:(self.knownboxes[end]+1)]
        usedmb = {k: self.matchboxes[k] for k in usedmbkeys}
        usednights = self.nights[:(self.knownnights[end]+1)]
        usedknownpm = list(set(usedmb.keys()) & set(self.knownpm))
        usedhaspm = [e for p in usedknownpm for e in p]
        usedbo = [bo for bo in self.bonights if bo <= self.knownnights[end]]

        return usedmb, usednights, usedknownpm, usedhaspm, usedbo

    def no_match(self, l: str, r: str, end) -> bool:
        '''(l,r) are definitely no match'''
        assert l in self.lefts and r in self.rights, f"f: {l}, s: {r}"

        mb, nights, kpm, hpm, bon = self.get_upto_episode(end)

        # one is part of known perfect match
        # matchbox result was false
        # pair in blackout night who is not known perfect match
        return ((l, r) not in kpm and ((l in hpm) or (r in hpm))) \
            or ((l, r) in mb.keys() and not mb[(l, r)]) \
            or any([(l, r) in nights[bo][0] and (l, r) not in kpm for bo in bon])

    def get_possible_matches(self, end):
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

    def solution_correct_format(self, solution: Set[Tuple[str, str]]) -> bool:
        if len(solution) != 11:
            print(solution)
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
            print(leftseated)
            print(rightseated)
            print(solution)
            return False
        return True

    def solution_possible(self, solution: Set[Tuple[str, str]], end) -> bool:
        assert self.solution_correct_format(solution)

        mb, nights, kpm, hpm, bon = self.get_upto_episode(end)

        # known perfect matches are in sol
        if not all([pm in solution for pm in kpm]):
            return False
        #  no known nomatches  are in sol
        elif any([p for p in solution if self.no_match(*p, end)]):
            return False

        if self.dmtuple is not None:
            # perfect matches of dbpair, must be the same person
            dml = [l for (l, r) in solution if r in self.dmtuple]
            if dml[0] != dml[1]:
                return False

        for pairs, lights in nights:
            intersection = set(pairs) & solution
            if len(intersection) != lights:
                return False
        return True

    def generate_solutions(self, end):
        '''Generating solutions that fit matchboxes, blackout nights and possible double matches'''
        def zip_product(ordering):
            return list(zip(self.lefts, ordering))

        possible_matches = self.get_possible_matches(end)

        mb, nights, kpm, hpm, bon = self.get_upto_episode(end)

        if self.dm is not None:
            # lefts who can have a double match
            dm_lefts = [
                l for l in self.lefts if not self.no_match(l, self.dm, end)]

            # remove dm right from possible matches to avoid duplicates solution generation
            modified_pos_matches = {l: [r for r in possible_matches[l]
                                        if r != self.dm or (l, r) in kpm]
                                    for l in possible_matches}

            # get first ten matches
            products = [list(ps) for ps in itertools.product(*modified_pos_matches.values())
                        if len(set(ps)) == 10]
            solutions_tenmatches = list(map(zip_product, products))

            solutions = []
            if self.dmtuple is None:
                pos_additional_matches = [(l, self.dm) for l in dm_lefts]

                for tenmatches, additionalmatch in itertools.product(solutions_tenmatches, pos_additional_matches):
                    if additionalmatch in tenmatches:
                        print(additionalmatch in tenmatches)
                        continue

                    solutions.append(set(tenmatches + [additionalmatch]))

            else:
                # In VIP 2023, we know the pair with
                dm1, dm2 = self.dmtuple
                if dm1 == self.dm:
                    dm1, dm2 = dm2, dm1

                for tenmatches in solutions_tenmatches:
                    dml = [l for (l, r) in tenmatches if r == dm1]
                    sol = set(tenmatches+[(dml[0], dm2)])
                    solutions.append(sol)
        else:
            modified_pos_matches = possible_matches
            for l in modified_pos_matches:
                print(l, modified_pos_matches[l])

             # get first ten matches
            products = [list(ps) for ps in itertools.product(*modified_pos_matches.values())
                        if len(set(ps)) == 10]
            solutions_tenmatches = list(map(zip_product, products))

            solutions = []

            # dm_rights = [r for r in self.rights if r not in hpm]
            # dm_lefts = list(
            #     filter(lambda x: x not in hpm, self.lefts))

        return solutions

    def find_solutions(self, end):
        solutions = self.generate_solutions(end)

        solutions = list(
            filter(lambda s: self.solution_possible(s, end), solutions))

        return solutions

    def analysize_solutions(self, fn, dbpair: Union[Tuple[str, str], None] = None):
        with open(fn, "r") as f:
            currentsols = eval(f.readlines()[-1])

        possols = list(filter(self.solution_possible, currentsols))

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
        fields = [""] + self.rights
        rows = [[l] + [str(pms[(l, r)]) if (l, r) in pms else "0" for r in self.rights]
                for l in self.lefts]
        df = pandas.DataFrame(data=rows, columns=fields)
        print(df)

        return


if __name__ == "__main__":
    allseasons = ["normalo2020", "normalo2021", "normalo2022", "normalo2023",
                  "vip2021", "vip2022", "vip2023"]
    import math

    # solnormalo2020 = {("Laura", "Aleks"), ("Luisa", "Axel"),("Ivana", "Dominic"), ("Madleine", "Edin"),
    #        ("Sabrina", "Elisha"), ("Nadine", "Ferhat"), ("Madleine", "Juliano"),
    #        ("Kathi", "Kevin"), ("Melissa", "Laurin"), ("Aline", "Mo"), ("Michelle", "René")}

    solnormalo2023 = {("Larissa", "Barkin"), ("Juliette", "Burim"), ("Steffi", "Cris"),
                      ("Carina", "Deniz"), ("Valeria", "Joel"), ("Caro", "Ken"),
                      ("Henna", "Kenneth"), ("Vanessa", "Marwin"), ("Caro", "Max"),
                      ("Aurelia", "Pascal"), ("Dorna", "Sasa")}

    for seasonfn in allseasons[3:4]:
        with open(f"data/{seasonfn}.json", "r") as f:
            seasondata = json.loads(f.read())

        print(seasonfn)
        season = AYTO(seasondata)

        res = season.find_solutions(9)
        print(res, len(res))
        # print("this", season.solution_possible(solnormalo2023, 9))
        # res = season.generate_solutions(9)

        # x = 1
        # for i, (p, n) in enumerate(season.nights):
        #     x *= math.comb(10, n)
        #     print(f"{i}: {n} lights, {math.comb(10,n)}")
        # print(x)

    # normalo22: 1258416
