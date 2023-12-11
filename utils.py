import json
import pandas as pd
from typing import List, Tuple, Dict, Set, Union, Optional
from ayto import AYTO


def check_nights(nights: List, lefts: List, rights: List) -> bool:
    '''Check if nights are fine'''

    # make sure all pairs are valid pairs and there are no duplicates
    for pairs, lights in nights:
        seated_lefts, seated_rights = [], []

        for l, r in pairs:
            if l in lefts and r in rights:
                # check that we don't have double seatings
                if l in seated_lefts or r in seated_rights:
                    print(f"{l} or {r} has already been seated")
                    return False
                seated_lefts.append(l)
                seated_rights.append(r)
            elif l in rights and r in lefts:
                # check that we don't have double seatings
                if l in seated_rights or r in seated_lefts:
                    print(f"{l} or {r} has already been seated")
                    return False
                seated_lefts.append(r)
                seated_rights.append(l)
            else:
                print(f"{l} or {r} is neither in the list of women or men")
                return False
        if not 0 <= lights <= 10:
            print(f"Number of lights not possible")
            return False
    return True


def read_data(fn: str) -> AYTO:
    with open(f"data/{fn}.json", "r") as f:
        jsondata: Dict = json.loads(f.read())

    assert "women" in jsondata and "men" in jsondata

    women = jsondata["women"]
    men = jsondata["men"]
    nummatches = max(len(women), len(men))

    # set lefts and rights
    if len(women) == nummatches and len(men) == 10:
        lefts, rights = men, women
    elif len(women) == 10 and len(men) == nummatches:
        lefts, rights = women, men
    else:
        raise ValueError(f"Not enough or too much women or men")

    # reading nights: seatings and lights for nights
    nightsdf = pd.read_excel(
        f"data/{fn}.xlsx", sheet_name="Nights",
        header=0, index_col=0)
    leftsh = nightsdf.columns.values.tolist()
    nights = [(list(zip(leftsh, row[:-1])), row[-1])
              for row in nightsdf.values.tolist()]
    # check nights
    assert check_nights(nights, lefts, rights)

    # matchboxes with results
    assert "matchboxes" in jsondata, "Data does not have matchboxes key"
    matchboxes = {}
    for (l, r), result in jsondata["matchboxes"]:
        if (l, r) in matchboxes or (r, l) in matchboxes:
            raise AssertionError(
                f"{(l,r)} occurs more than once in matchboxes")

        if l in lefts and r in rights:
            matchboxes[(l, r)] = result
        elif l in rights and r in lefts:
            matchboxes[(r, l)] = result
        else:
            print(l,r)
            print(jsondata["matchboxes"])
            raise AssertionError(f"{l} or {r} is not a valid name")

    # nights with blackout
    bonights = jsondata["bonights"]

    # do we know who is the second/third match to someone
    # besides normalo2023, yes
    dm = jsondata.get("dm", None)
    tm = jsondata.get("tm", None)
    assert (dm is None or dm in rights) and (tm is None or tm in rights)
    # do we know which two share the same match
    # besides vip2023, no
    dmtuple = jsondata.get("dmtuple", None)
    assert dmtuple is None or (dmtuple[0] in rights and dmtuple[1] in rights), \
        f"{dmtuple} may have spelling error"

    # matching night that was cancelled
    cancellednight = jsondata.get("cancelled", -1)
    assert -1 <= 0 <= 9, f"Invalid value for cancelled night"

    # mapping from matchboxes to episode
    boxesepisodes = jsondata.get("boxesepisode", list(range(10)))
    assert len(boxesepisodes) == len(matchboxes), f"{len(boxesepisodes),len(matchboxes)}"

    # solution
    solution = {(l, r) for l, r in jsondata.get("solution", [])}

    return AYTO(lefts=lefts, rights=rights,
                nights=nights, matchboxes=matchboxes,
                bonights=bonights,
                dm=dm, dmtuple=dmtuple, tm=tm,
                cancellednight=cancellednight,
                boxesepisodes=boxesepisodes,
                solution=solution)


def analysize_solutions(season: AYTO, fn: str, end: int) -> None:
    with open(fn, "r") as f:
        content = f.readlines()

    print(f"i: {end} \n")

    mbs = season.get_matchboxes(end)
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

    mostlikely = list(pdict.items())[:season.nummatches]

    print(f"{mostlikely}\n")

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


