from collections.abc import Sequence

import pandas as pd

from .ayto import Solver
from .aytonormalo24 import Normalo2024Solver
from .aytovip23 import VIP2023Solver
from .aytovip25 import VIP2025Solver
from .aytovip26 import VIP2026Solver
from .models import *


def check_nights(
    nights: tuple[Night, ...], lefts: tuple[str, ...], rights: tuple[str, ...]
) -> None:
    """Check if nights are fine"""

    # make sure all pairs are valid pairs and there are no duplicates
    for i, night in enumerate(nights):  # pyright: ignore[reportUnusedVariable]
        seated_lefts: list[str] = []
        seated_rights: list[str] = []

        for l, r in night.pairs:
            if l in lefts and r in rights:
                # check that we don't have double seatings
                if l in seated_lefts or r in seated_rights:
                    raise ValueError(
                        f"Either {l} or {r} has already been seated  in night {night}"
                    )
                seated_lefts.append(l)
                seated_rights.append(r)
            elif l in rights and r in lefts:
                raise ValueError(
                    f"Pair {(l, r)} is in the wrong order in night {night}"
                )
            else:
                raise ValueError(f"{l} or {r} is neither in the list of women or men")
        if not 0 <= night.lights <= 10:
            raise ValueError(f"Number of lights not possible in night {night}")


def validate_season_args(
    lefts: tuple[str, ...],
    rights: tuple[str, ...],
    nights: tuple[Night, ...],
    matchboxes: Matchboxes,
    dm: str | None,
):

    nummatches = max(len(lefts), len(rights))
    # todo: remove Laurenz as hardcoded value
    if "Laurenz" not in lefts and (len(lefts) != 10 or len(rights) != nummatches):
        raise ValueError("Not enough or too much women or men")

    # check nights
    check_nights(nights, lefts, rights)

    # matchboxes with results
    pairs: set[Pair] = set()
    for _, (l, r), _, _ in matchboxes:
        if (l, r) in pairs or (r, l) in pairs:
            raise ValueError(f"{(l, r)} occurs more than once in matchboxes")
        if l in lefts and r in rights:
            pairs.add((l, r))
        elif l in rights and r in lefts:
            raise ValueError(f"Pair {(l, r)} is in the wrong order in matchboxes")
        else:
            raise ValueError(f"{l} or {r} is not a valid name")

    if dm and dm in lefts:
        raise ValueError(f"{dm} is part of lefts (the smaller gender group)")


def make_pair_list(lefts: Sequence[str], rights: Sequence[str]) -> tuple[Pair, ...]:
    if len(lefts) != len(rights):
        raise ValueError("Cannot make list of pairs out of lists of different length")
    return tuple((l, r) for l, r in zip(lefts, rights))


def read_data_from_excel(
    sn: str,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[Night, ...],
    Matchboxes,
    str | None,
    Solution,
]:

    dfs = pd.read_excel(f"data/{sn}.xlsx", sheet_name=None, header=0)
    dfcand: pd.DataFrame = dfs["Candidates"]
    lefts: tuple[str, ...] = tuple(dfcand["left"].dropna().tolist())
    rights: tuple[str, ...] = tuple(dfcand["right"].dropna().tolist())
    dmlist = dfcand["mm"].dropna().tolist() if "mm" in dfcand.columns else []
    if len(dmlist) > 1:
        raise ValueError(f"More than one DM in Candidates sheet for {sn}")
    dm: str | None = dmlist[0] if len(dmlist) == 1 else None
    if sn == "normalo2024":
        dm = dfcand["mm"].dropna().tolist()[0] if "mm" in dfcand.columns else None

    dfnights = dfs["Nights"]
    nights: tuple[Night, ...] = ()
    if sn != "vip2026":
        nights = tuple(
            Night(
                set(make_pair_list(list(dfnights.columns[:-1]), row[:-1])),
                int(row[-1]),  # pyright: ignore[reportAny]
            )
            for row in dfnights.values.tolist()
        )
    else:
        pairspnight = [
            make_pair_list(list(dfnights.columns[:-1]), row[:-1])
            for row in dfnights.values.tolist()
        ]
        pairspnight = [
            tuple(filter(lambda x: not pd.isna(x[1]), pairs)) for pairs in pairspnight
        ]
        nights = tuple(
            Night(
                set(pairspnight[i]),
                int(dfnights["lights"].iloc[i]),  # pyright: ignore[reportAny]
            )
            for i in range(dfnights.shape[0])
        )

    dfmatchboxes = dfs["Matchboxes"]
    matchboxes: Matchboxes = Matchboxes(
        tuple(dfmatchboxes["episode"].to_list()),
        make_pair_list(
            dfmatchboxes.iloc[:, 1].to_list(), dfmatchboxes.iloc[:, 2].to_list()
        ),
        tuple(dfmatchboxes["result"].to_list()),
        bfs=tuple(dfmatchboxes["before_night"].to_list()),
    )

    validate_season_args(lefts, rights, nights, matchboxes, dm)
    solution: Solution = Solution()
    dfsolution = dfs.get("Solution")
    if dfsolution is not None:
        solution = Solution(
            make_pair_list(dfsolution["left"].to_list(), dfsolution["right"].to_list())
        )
    else:
        solution = Solution()

    return lefts, rights, nights, matchboxes, dm, solution


def sols_as_df(sols: list[Solution]) -> pd.DataFrame:
    rows = [{r: l for (l, r) in s} for s in sols]
    df = pd.DataFrame(rows)
    return df


def get_season(sn: str) -> Season:
    lefts, rights, nights, matchboxes, mm, solution = read_data_from_excel(sn)
    args = (sn, lefts, rights, nights, matchboxes, solution, mm)
    if sn == "normalo2024":
        season = Season(*args, num_matches=12, max_mm_size=3)
    elif sn == "normalo2026":
        season = Season(*args, mm_known_after_week=5, mb_reveals_dm=False)
    elif sn == "vip2023":
        season = Season(
            *args,
            double_match_pair=("Peter", "Max"),
        )
    elif sn == "vip2025":
        season = Season(*args, num_matches=12, has_two_dms=True)
    elif sn == "vip2026":
        season = Season(*args, num_matches=12)
    else:
        season = Season(*args)
    return season


def get_solver(sn: str) -> Solver:
    season = get_season(sn)
    if sn == "normalo2024":
        solver = Normalo2024Solver(season)
    elif sn == "vip2023":
        solver = VIP2023Solver(season)
    elif sn == "vip2025":
        solver = VIP2025Solver(season)
    elif sn == "vip2026":
        solver = VIP2026Solver(season)
    else:
        solver = Solver(season)
    return solver
