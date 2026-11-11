from collections.abc import Iterable
from typing import Literal, overload

import pandas as pd

from ayto.bitmask_solver import BitMaskSolver
from ayto.double_match_for_right_solver import (
    DoubleMatchForRightBitMaskSolver,
    DoubleMatchForRightSolver,
)
from ayto.double_match_pair_solver import (
    DoubleMatchPairBitMaskSolver,
    DoubleMatchPairSolver,
)
from ayto.season import (
    Matchboxes,
    Night,
    Pair,
    Person,
    Season,
    Solution,
    matchboxes_from_list,
)
from ayto.solver import TupleSolver
from ayto.triple_match_solver import TripleMatchBitMaskSolver, TripleMatchSolver
from ayto.two_double_matches_solver import (
    TwoDoubleMatchesBitMaskSolver,
    TwoDoubleMatchesSolver,
)


def validate_nights(
    nights: list[tuple[list[str], list[str], int]],
    lefts: tuple[str, ...],
    rights: tuple[str, ...],
) -> None:
    """Check if nights are fine"""

    # make sure all pairs are valid pairs and there are no duplicates
    for i, (plefts, prights, lights) in enumerate(nights):
        seated_lefts: list[str] = []
        seated_rights: list[str] = []

        for l, r in zip(plefts, prights):
            if l in lefts and r in rights:
                # check that we don't have double seatings
                if l in seated_lefts or r in seated_rights:
                    raise ValueError(
                        f"Either {l} or {r} has already been seated  in night {i}"
                    )
                seated_lefts.append(l)
                seated_rights.append(r)
            elif l in rights and r in lefts:
                raise ValueError(f"Pair {(l, r)} is in the wrong order in night {i}")
            else:
                raise ValueError(f"{l} or {r} is neither in the list of women or men")
        if not 0 <= lights <= 10:
            raise ValueError(f"Number of lights not possible in night {i}")


def validate_matchboxes(
    matchboxes: list[tuple[int, tuple[str, str], bool, bool]],
    lefts: tuple[str, ...],
    rights: tuple[str, ...],
):
    pairs: set[tuple[str, str]] = set()
    for _, (l, r), _, _ in matchboxes:
        if (l, r) in pairs or (r, l) in pairs:
            raise ValueError(f"{(l, r)} occurs more than once in matchboxes")
        if l in lefts and r in rights:
            pairs.add((l, r))
        elif l in rights and r in lefts:
            raise ValueError(f"Pair {(l, r)} is in the wrong order in matchboxes")
        else:
            raise ValueError(f"{l} or {r} is not a valid name")


def make_pair_list(
    lefts: Iterable[Person], rights: Iterable[Person]
) -> tuple[Pair, ...]:
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

    dfs = pd.read_excel(f"data/seasons/{sn}.xlsx", sheet_name=None, header=0)
    dfcand: pd.DataFrame = dfs["Candidates"]
    lefts: tuple[str, ...] = tuple(dfcand["left"].dropna().tolist())
    rights: tuple[str, ...] = tuple(dfcand["right"].dropna().tolist())
    dmlist = dfcand["mm"].dropna().tolist() if "mm" in dfcand.columns else []
    if len(dmlist) > 1:
        raise ValueError(f"More than one DM in Candidates sheet for {sn}")

    dm: str | None = dmlist[0] if len(dmlist) == 1 else None
    if dm and dm in lefts:
        raise ValueError(f"{dm} is part of lefts (the smaller gender group)")

    dfnights = dfs["Nights"]
    nights: tuple[Night, ...] = ()
    night_args: list[tuple[list[str], list[str], int]] = []
    if sn != "vip2026":
        night_args = [
            (
                [str(x) for x in dfnights.columns[:-1]],
                [str(x) for x in row[:-1]],  # pyright: ignore[reportAny]
                int(row[-1]),  # pyright: ignore[reportAny]
            )
            for row in dfnights.values.tolist()
        ]
    else:
        pairspnight = [
            zip(list(dfnights.columns[:-1]), row[:-1])
            for row in dfnights.values.tolist()
        ]
        pairspnight = [
            tuple(
                filter(lambda x: not pd.isna(x[1]), pairs)  # pyright: ignore[reportAny]
            )
            for pairs in pairspnight
        ]
        night_args = [
            (
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                int(dfnights["lights"].iloc[i]),  # pyright: ignore[reportAny]
            )
            for i, pairs in enumerate(pairspnight)
        ]

    validate_nights(night_args, lefts, rights)

    dfmatchboxes = dfs["Matchboxes"]

    matchbox_args: list[tuple[int, tuple[str, str], bool, bool]] = [
        (row[0], (row[1], row[2]), row[3], row[4])
        for row in dfmatchboxes.values.tolist()
    ]

    validate_matchboxes(matchbox_args, lefts, rights)

    matchboxes = matchboxes_from_list(
        [
            (ep, (lefts.index(l), rights.index(r)), res, bf)
            for ep, (l, r), res, bf in matchbox_args
        ]
    )

    solution: Solution = Solution()
    dfsolution = dfs.get("Solution")
    if dfsolution is not None:
        solution = Solution(
            make_pair_list(
                map(lefts.index, dfsolution["left"]),
                map(rights.index, dfsolution["right"]),
            )
        )
    else:
        solution = Solution()

    nights = tuple(
        Night(
            frozenset(make_pair_list(map(lefts.index, ls), map(rights.index, rs))),
            light,
        )
        for ls, rs, light in night_args
    )

    return lefts, rights, nights, matchboxes, dm, solution


def sols_as_df(sols: list[Solution]) -> pd.DataFrame:
    rows = [{r: l for (l, r) in s} for s in sols]
    df = pd.DataFrame(rows)
    return df


def get_season(sn: str) -> Season:
    import json

    lefts, rights, nights, matchboxes, _, solution = read_data_from_excel(sn)
    args = (sn, lefts, rights, nights, matchboxes, solution)
    with open(f"data/season_configs/{sn}.json") as f:
        logic_args: dict[str, object] = json.load(f)  # pyright: ignore[reportAny]
    season = Season(*args, **logic_args)  # pyright: ignore[reportArgumentType]

    return season


@overload
def get_solver(sn: str, bitmask: Literal[True]) -> BitMaskSolver: ...


@overload
def get_solver(sn: str, bitmask: Literal[False] = False) -> TupleSolver: ...


def get_solver(sn: str, bitmask: bool = False) -> TupleSolver | BitMaskSolver:
    season = get_season(sn)
    if bitmask:
        match season.match_model:
            case "base":
                solver = BitMaskSolver(season)
            case "double_match_pair":
                solver = DoubleMatchPairBitMaskSolver(season)
            case "two_double_matches":
                solver = TwoDoubleMatchesBitMaskSolver(season)
            case "double_match_for_right":
                solver = DoubleMatchForRightBitMaskSolver(season)
            case "triple_match":
                solver = TripleMatchBitMaskSolver(season)
            case _:
                raise ValueError(
                    f"Unknown match model {season.match_model} for season {sn}"
                )
    else:
        match season.match_model:
            case "base":
                solver = TupleSolver(season)
            case "double_match_pair":
                solver = DoubleMatchPairSolver(season)
            case "two_double_matches":
                solver = TwoDoubleMatchesSolver(season)
            case "double_match_for_right":
                solver = DoubleMatchForRightSolver(season)
            case "triple_match":
                solver = TripleMatchSolver(season)
            case _:
                raise ValueError(
                    f"Unknown match model {season.match_model} for season {sn}"
                )

    return solver
