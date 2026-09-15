import pandas as pd
from typing import Sequence

from .models import *
from .aytonormalo24 import Normalo2024Solver
from .aytovip25 import VIP2025Solver
from .aytovip26 import VIP2026Solver
from .ayto import Solver




def check_nights(nights: tuple[Night,...], lefts: tuple[str,...], rights: tuple[str,...]) -> None:
    """Check if nights are fine"""

    # make sure all pairs are valid pairs and there are no duplicates
    ni = 0
    for night in nights:
        seated_lefts, seated_rights = [], []

        for l,r in night.pairs:
            if l in lefts and r in rights:
                # check that we don't have double seatings
                if l in seated_lefts or r in seated_rights:
                    raise ValueError(f"Either {l} or {r} has already been seated  in night {night}")
                seated_lefts.append(l)
                seated_rights.append(r)
            elif l in rights and r in lefts:
                raise ValueError(f"Pair {(l,r)} is in the wrong order in night {night}")
            else:
                raise ValueError(
                    f"{l} or {r} is neither in the list of women or men"
                )
        if not 0 <= night.lights <= 10:
            raise ValueError(f"Number of lights not possible in night {night}")
        ni += 1


def validate_season_args(
    lefts: tuple[str,...],
    rights: tuple[str,...],
    nights: tuple[Night,...],
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
    for n, (l,r), result in matchboxes:

        if (l,r) in pairs or (r, l) in pairs:
            raise ValueError(f"{(l,r)} occurs more than once in matchboxes")
        if l in lefts and r in rights:
            pairs.add((l,r))
        elif l in rights and r in lefts:
            raise ValueError(f"Pair {(l,r)} is in the wrong order in matchboxes")
        else:
            raise ValueError(f"{l} or {r} is not a valid name")

    if dm and dm in lefts:
        raise ValueError(f"{dm} is part of lefts (the smaller gender group)")

    return


def make_pair_list(lefts: Sequence[str], rights: Sequence[str]) -> tuple[Pair, ...]:
    if len(lefts) != len(rights):
        raise ValueError("Cannot make list of pairs out of lists of different length")
    return tuple((l, r) for l, r in zip(lefts, rights))


def read_data_from_excel(
    sn: str,
) -> tuple[tuple[str,...], tuple[str,...], tuple[Night,...], Matchboxes, str | None, Solution]:
    dfcand = pd.read_excel(f"data/{sn}.xlsx", sheet_name="Candidates", header=0)
    lefts: tuple[str] = tuple(dfcand["left"].dropna().tolist())
    rights: tuple[str] = tuple(dfcand["right"].dropna().tolist())
    dmlist = dfcand["mm"].dropna().tolist() if "mm" in dfcand.columns else []
    if len(dmlist) > 1:
        raise ValueError(f"More than one DM in Candidates sheet for {sn}")
    dm: str | None = dmlist[0] if len(dmlist) == 1 else None
    if sn == "normalo2024":
        dm = dfcand["mm"].dropna().tolist()[0] if "mm" in dfcand.columns else None
    dfnights = pd.read_excel(f"data/{sn}.xlsx", sheet_name="Nights", header=0)
    nights: tuple[Night, ...] = tuple(
        Night(make_pair_list(list(dfnights.columns[:-1]), row[:-1]), int(row[-1]))
        for row in dfnights.values.tolist()
    )

    dfmatchboxes = pd.read_excel(f"data/{sn}.xlsx", sheet_name="Matchboxes", header=0)
    matchboxes: Matchboxes = Matchboxes(
        tuple(dfmatchboxes["episode"].to_list()),
        make_pair_list(dfmatchboxes["left"].to_list(), dfmatchboxes["right"].to_list()),
        tuple(dfmatchboxes["result"].to_list()),
    )

    validate_season_args(lefts, rights, nights, matchboxes, dm)
    
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


def product_without_reps(arr: list[list]):
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




def get_season(sn: str) -> Season:
    lefts, rights, nights, matchboxes, mm, solution = read_data_from_excel(sn)
    if sn == "normalo2024":
        season = Season(sn, lefts, rights, nights, matchboxes, solution=solution, mm=mm)
    elif sn == "vip2023":
        # return
        season = Season(
            sn,
            lefts,
            rights,
            nights,
            matchboxes,
            solution=solution,
            mm=mm,
            dmtuple=("Peter", "Max"),
        )
    elif sn == "vip2025":
        season = Season(
            sn,
            lefts,
            rights,
            nights,
            matchboxes,
            solution=solution,
            mm=mm,
            two_dms=True,
        )
    elif sn == "vip2026":
        season: Season = Season(
            sn, lefts, rights, nights, matchboxes, solution=solution
        )
    else:
        season: Season = Season(
            sn, lefts, rights, nights, matchboxes, solution=solution, mm=mm
        )
    return season


def get_solver(sn:str) -> Solver:
    season = get_season(sn)
    if sn == "normalo2024":
        solver = Normalo2024Solver(season)
    elif sn == "vip2025":
        solver = VIP2025Solver(season)
    elif sn == "vip2026":
        solver = VIP2026Solver(season)
    else:
        solver = Solver(season)
    return solver




