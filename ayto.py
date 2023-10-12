from typing import List, Tuple, Dict, Set, Union
import itertools
from copy import deepcopy
import scipy
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

        if "dmtuple" in data:
            dm1, dm2 = data["dmtuple"]
            if not (dm1 in self.rights and dm2 in self.rights):
                raise ValueError(
                    f"{dm1} or {dm2} may be misspelled for dmtuple")
            self.dmtuple = (dm1, dm2)
        else:
            self.dmtuple = None

        # seatings and lights for nights
        self.read_nights()

        # matchboxes with results
        self.read_matchboxes()

        # nights with blackout
        self.bonights = data["bonights"]

        # known perfect matches
        self.knownpm = [p for p in self.matchboxes.keys()
                        if self.matchboxes[p]]

        # people who have known perfect match
        self.haspm = [e for p in self.knownpm for e in p]

        # possible matches for each person without considering lights in matching nights
        self.possible_matches = {l: [r for r in self.rights if not self.no_match(l, r)]
                                 for l in self.lefts}

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

    def no_match(self, l: str, r: str) -> bool:
        '''(l,r) are definitely no match'''
        assert l in self.lefts and r in self.rights, f"f: {l}, s: {r}"

        # one is part of known perfect match
        # matchbox result was false
        # pair in blackout night who is not known perfect match
        return ((l, r) not in self.knownpm and ((l in self.haspm) or (r in self.haspm))) \
            or ((l, r) in self.matchboxes.keys() and not self.matchboxes[(l, r)]) \
            or any([(l, r) in self.nights[bo][0] and (l, r) not in self.knownpm for bo in self.bonights])

    def solution_correct_format(self, solution: Set[Tuple[str, str]]) -> bool:
        '''documentation fehlt :pepegun: mach endlich mal was produktives isa'''
        if len(solution) != 11:

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

        return 0 not in leftseated.values() and 0 not in rightseated.values()

    def solution_possible(self, solution: Set[Tuple[str, str]]) -> bool:
        assert self.solution_correct_format(solution)
        # known perfect matches are in sol
        if not all([pm in solution for pm in self.knownpm]):
            return False
        #  no known nomatches  are in sol
        elif any([p for p in solution if self.no_match(*p)]):
            return False

        if self.dmtuple is not None:
            # perfect matches of dbpair, must be the same person
            dml = [l for (l, r) in solution if r in self.dmtuple]
            if dml[0] != dml[1]:
                return False

        for pairs, lights in self.nights:
            intersection = set(pairs) & solution
            if len(intersection) != lights:
                return False
        return True

    def find_solutions(self, def_double_match: bool = True):
        '''def_double_match: bool, if true, we know that the last of the rights is a double match to someone'''

        if def_double_match:
            dm_rights = [self.rights[-1]]
            dm_lefts = [l for l in self.lefts if not self.no_match(
                l, self.rights[-1])]
            # remove last right from possible matches to avoid duplicates
            modified_pos_matches = {l: [r for r in self.possible_matches[l]
                                        if r not in dm_rights]
                                    for l in self.possible_matches}
        else:
            dm_rights = list(
                filter(lambda x: x not in self.haspm, self.rights))
            dm_lefts = list(
                filter(lambda x: x not in self.haspm, self.lefts))
            modified_pos_matches = self.possible_matches

        # get first ten matches

        def zip_product(ordering):
            return list(zip(self.lefts, ordering))
        products = [list(ps) for ps in itertools.product(*modified_pos_matches.values())
                    if len(set(ps)) == 10]
        solutions_tenmatches = list(map(zip_product, products))

        # print(f"solutions_tenmatches: {len(solutions_tenmatches)}")

        # add additional eleventh match
        solutions = []
        if self.dmtuple is None:
            pos_additional_matches = list(
                itertools.product(dm_lefts, dm_rights))
            for tenmatches, additionalmatch \
                    in list(itertools.product(solutions_tenmatches, pos_additional_matches)):
                solutions.append(set(tenmatches + [additionalmatch]))
        else:
            dm1, dm2 = self.dmtuple
            if dm1 in dm_rights:
                dm1, dm2 = dm2, dm1

            for tenmaches in solutions_tenmatches:
                dml = [l for (l, r) in tenmaches if r == dm1]
                sol = set(tenmaches+[(dml[0], dm2)])
                solutions.append(sol)

        # print(f"solutions: {len(solutions)}")
        # find all possible solutions
        solutions = list(
            filter(self.solution_possible, solutions))

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


def transform_dictionnary(mydict: Dict):
    lkeys, rkeys = list(zip(*mydict.keys()))
    lkeys, rkeys = set(lkeys), set(rkeys)
    x = {l: {r: mydict[(l, r)] for r in rkeys} for l in lkeys}
    return x


if __name__ == "__main__":
    with open("data/vip2022.json", "r") as f:
        seasondata = json.loads(f.read())

    season = AYTO(seasondata)
    res = season.find_solutions()
    print(res, len(res))

    # TODO: add known right dm to data
    # TODO: add cancelled nights