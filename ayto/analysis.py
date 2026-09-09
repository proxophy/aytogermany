
import pandas as pd

from .ayto import *

from .solver import find_solutions
from .ayto import AYTO, PartialSol, CompleteSol
from .utils import time_it



@time_it
def analysize_solutions(season: AYTO, options: dict):
    mbs = season.get_matchboxes(options)
    sols = find_solutions(season, options)
    end = options.get("end", season.numepisodes - 1)
    end = min(end, season.numepisodes - 1)

    pairs_counter = Counter([p for s in sols for p in s])

    allpairs = itertools.product(season.lefts, season.rights)
    perfect_matches = [p for p in pairs_counter if pairs_counter[p] == len(sols)]
    new_nomatches = list(
        filter(
            lambda p: not (season.no_match(*p, options) or p in pairs_counter), allpairs
        )
    )
    new_pms = list(filter(lambda p: p not in mbs, perfect_matches))
    # todo: max end with num episodes
    print(f"Nach Folgen {2*end+3} & {2*end+4}")
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
    dm_lefts = [
        (l, round(r / len(sols) * 100, 1))
        for (l, r) in Counter([dm_left(s) for s in sols]).items()
    ]
    dm_lefts.sort(key=(lambda a: a[1]), reverse=True)
    print(f"Person mit Doppelmatch: {dm_lefts}")

    data = {
        l: pd.Series(
            [
                round(pairs_counter.get((l, r), 0) / len(sols) * 100, 1)
                for r in season.rights
            ],
            index=season.rights,
        )
        for l in season.lefts
    }
    df = pd.DataFrame(data)

    return df


def matching_night_probs(season: AYTO, episode: int):
    options = {"end": episode, "includenight": False, "verbose": False}
    beforenight = find_solutions(season, options)
    night = season.get_nights({"end": episode, "includenight": True})[-1][0]
    nightpossol = any([set(night).issubset(sol) for sol in beforenight])

    print(f"Pairs of nights are possible solution: {nightpossol}")

    poslights = Counter([len(set(night).intersection(sol)) for sol in beforenight])
    return [
        (i, round(poslights.get(i, 0) / len(beforenight) * 100, 2)) for i in range(0, 11)
    ]



def sol_probs(sols: list[CompleteSol], sol: CompleteSol, options: dict):
    # sols = find_solutions(season, options)
    nightpossol = any([set(sol).issubset(sol) for sol in sols])
    
    # print(f"Solution possible at this point: {nightpossol}")
    lights = Counter([len(s & sol) for s in sols])
    probs = [
        round(lights.get(i, 0) / len(sols) * 100, 2) for i in range(12+ 1)
    ]
    import statistics

    return probs, statistics.median(filter(lambda x: x > 0,probs )), max(filter(lambda x: x > 0,probs ))