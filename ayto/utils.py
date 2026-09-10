import pandas as pd
import functools
import time
from typing import Sequence

from .models import *


def time_it(inner):
    @functools.wraps(inner)
    def c_inner(*args):
        start = time.time()
        res = inner(*args)
        end = time.time()
        print(f"=== time needed for {inner.__name__}: {(end-start):0.3f}s ===")
        return res

    return c_inner


def check_nights(nights: list[Night], lefts: list[str], rights: list[str]) -> None:
    """Check if nights are fine"""

    # make sure all pairs are valid pairs and there are no duplicates
    ni = 0
    for night in nights:
        seated_lefts, seated_rights = [], []

        for p in night.pairs:
            if p.l in lefts and p.r in rights:
                # check that we don't have double seatings
                if p.l in seated_lefts or p.r in seated_rights:
                    raise ValueError(f"{p} has already been seated  in night {night}")
                seated_lefts.append(p.l)
                seated_rights.append(p.r)
            elif p.l in rights and p.r in lefts:
                raise ValueError(f"Pair {p} is in the wrong order in night {night}")
            else:
                raise ValueError(
                    f"{p.l} or {p.r} is neither in the list of women or men"
                )
        if not 0 <= night.lights <= 10:
            raise ValueError(f"Number of lights not possible in night {night}")
        ni += 1


def validate_season_args(
    lefts: list[str],
    rights: list[str],
    nights: list[Night],
    matchboxes: Matchboxes,
    dm: str | None,
):

    nummatches = max(len(lefts), len(rights))
    if "Laurenz" not in lefts and (len(lefts) != 10 or len(rights) != nummatches):
        raise ValueError(f"Not enough or too much women or men")

    # check nights
    check_nights(nights, lefts, rights)

    # matchboxes with results
    pairs = set()
    for n, p, result in matchboxes:

        if p in pairs or (p.r, p.l) in pairs:
            raise AssertionError(f"{p} occurs more than once in matchboxes")
        if p.l in lefts and p.r in rights:
            pairs.add(p)
        elif p.l in rights and p.r in lefts:
            raise ValueError(f"Pair {p} is in the wrong order in matchboxes")
        else:
            raise ValueError(f"{p.l} or {p.r} is not a valid name")

    if dm and dm in lefts:
        raise ValueError(f"{dm} is part of lefts (the smaller gender group)")

    return


def make_pair_list(lefts: Sequence[str], rights: Sequence[str]) -> tuple[Pair, ...]:
    if len(lefts) != len(rights):
        raise ValueError("Cannot make list of pairs out of lists of different length")
    return tuple(Pair(l, r) for l, r in zip(lefts, rights))


def read_data_from_excel(
    sn: str,
) -> tuple[list[str], list[str], list[Night], Matchboxes, str | None, Solution]:
    dfcand = pd.read_excel(f"data/{sn}.xlsx", sheet_name="Candidates", header=0)
    lefts: list[str] = dfcand["left"].dropna().tolist()
    rights: list[str] = dfcand["right"].dropna().tolist()
    dmlist = dfcand["mm"].dropna().tolist() if "mm" in dfcand.columns else []
    if len(dmlist) > 1:
        raise ValueError(f"More than one DM in Candidates sheet for {sn}")
    dm: str | None = dmlist[0] if len(dmlist) == 1 else None
    if sn == "normalo2024":
        print("reading normalo2024")
        dm = dfcand["mm"].dropna().tolist()[0] if "mm" in dfcand.columns else None
        print("tm", dm)
    dfnights = pd.read_excel(f"data/{sn}.xlsx", sheet_name="Nights", header=0)
    nights: list[Night] = [
        Night(make_pair_list(list(dfnights.columns[:-1]), row[:-1]), int(row[-1]))
        for row in dfnights.values.tolist()
    ]

    dfmatchboxes = pd.read_excel(f"data/{sn}.xlsx", sheet_name="Matchboxes", header=0)
    matchboxes: Matchboxes = Matchboxes(
        dfmatchboxes["episode"].to_list(),
        make_pair_list(dfmatchboxes["left"].to_list(), dfmatchboxes["right"].to_list()),
        dfmatchboxes["result"].to_list(),
    )

    validate_season_args(lefts, rights, nights, matchboxes, dm)
    # todo: type annotate and find out what set is
    try:
        dfsolution = pd.read_excel(f"data/{sn}.xlsx", sheet_name="Solution", header=0)
        solution: Solution = Solution(
            frozenset(
                make_pair_list(
                    dfsolution["left"].to_list(), dfsolution["right"].to_list()
                )
            )
        )
    except:
        solution: Solution = Solution()

    return lefts, rights, nights, matchboxes, dm, solution


def sols_as_df(sols) -> pd.DataFrame:
    rows = [{l: r for (l, r) in s} for s in sols]
    df = pd.DataFrame(rows)
    return df


def product_without_reps(arr: list[list]) :
    used = set()
    current = []

    def rec(i):
        if i == len(arr):
            yield current.copy()
            return

        for x in arr[i]:
            if x in used:
                continue

            used.add(x)
            current.append(x)

            yield from rec(i + 1)

            current.pop()
            used.remove(x)

    yield from rec(0)


def listeq(l1, l2):
    for l in l1:
        if l not in l2:
            return False
    for l in l2:
        if l not in l1:
            return False
    return True


if __name__ == "__main__":
    allseasons = [
        "normalo2020",
        "normalo2021",
        "normalo2022",
        "normalo2023",
        "normalo2024",
        "normalo2025",
        "normalo2026",
        "vip2021",
        "vip2022",
        "vip2023",
        "vip2024",
        "vip2025",
        "vip2026",
    ]

    res = read_data_from_excel("vip2022")
    for x in res:
        print(x)
