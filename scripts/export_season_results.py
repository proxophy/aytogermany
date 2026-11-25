# pyright: standard
import json
import os

import pandas as pd
from pandas.core.frame import DataFrame

from ayto import SolutionSpace, df_for_probs, get_season, get_solver
from ayto.base import SolutionType, Solver


def get_stats[S: SolutionType, T](
    solver: Solver[S, T], end: int
) -> tuple[DataFrame, int]:
    sols = solver.solve(end, True)
    print(f"sn = {solver.season.name}, end = {end}: #solutions = {len(sols)}")
    solsp = SolutionSpace(solver, sols)
    df = df_for_probs(solsp)
    return df, len(sols)


def export_season_results(sn: str):
    solver = get_solver(sn, True)
    episodes = {}
    for i in range(solver.season.num_weeks):
        df, num_sols = get_stats(solver, i)
        episodes[i + 1] = {
            "num_sols": num_sols,
            "probs": df.to_dict("split"),
        }
    data = {"episodes": episodes}  # pyright: ignore[reportUnknownVariableType]
    with open(f"analytics/probs/{sn}.json", "w") as f:
        json.dump(data, f)


def nights_to_df(sn: str):
    season = get_season(sn)
    data = []
    for i, night in enumerate(season.nights):
        nd: dict[str, int | str] = {"Week": i + 1}
        for l, r in night.pairs:
            nd[season.decode_left(l)] = season.decode_right(r)
        nd["Lights"] = night.lights
        data.append(nd)

    df = pd.DataFrame(data)
    print(df)
    df.to_csv(f"analytics/nights/{sn}.csv", index=False)


def export_excel(sn: str):
    d = pd.read_excel(f"data/seasons/{sn}.xlsx", sheet_name=None)
    print(d.keys())
    os.mkdir(f"data/csv/{sn}")
    for k, v in d.items():
        v.to_csv(f"data/csv/{sn}/{k.lower()}.csv", index=False)


if __name__ == "__main__":
    # uv run python -m scripts.export_season_results

    sn = "vip2021"
    # for sn in allseasons:
    # export_season_results(sn)
    export_excel(sn)
    # solver = get_solver(sn)
    # for sn in ALL_SEASONS:
    #     export_excel(sn)

    # df, _ = get_stats(solver, 6)
    # nights_to_df(sn)
    # print(df.to_dict("records"))
