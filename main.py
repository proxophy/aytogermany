
import pandas as pd
from collections import Counter

from ayto import Solver
from ayto.analysis import plot_probs
from ayto.models import Season, Solution, Pair
from ayto.utils import get_season, get_solver


def appendsol(sn: str, tsol: set[tuple[str, str]]):
    plefts, prights = zip(*tsol)
    dfsol = pd.DataFrame(data={"left": plefts, "right": prights})
    dfsol = dfsol.sort_values(by=["left", "right"]).reset_index(drop=True)
    print(dfsol)
    with pd.ExcelWriter(
        f"data/{sn}.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        dfsol.to_excel(writer, sheet_name="Solution", index=False)


def comparetoao(sols, end):
    with open("aodata.txt", "r") as f:
        lines = map(eval, f.readlines())

    def parse_sol(ld):
        pairs = set()
        for l, rs in ld.items():
            for r in rs:
                pairs.add((l, r))
        return Solution(frozenset(pairs))

    solsao = list(map(parse_sol, lines))

    print("lengths", len(solsao), len(sols))
    count = count2 = 0
    reasons = []
    # for s in solsao:
    #     spd = solver.solution_possible(s, end)
    #     if not spd["res"]:
    #         # print(spd["reason"], spd.get("detail", ""))

    #         count += 1
    #     elif s not in sols:
    #         count2 += 1
    #         print("s not in sols")
    print("count", count, count2)

    count = count2 = 0
    nomatchp = []
    for s in sols:
        if s not in solsao:
            print(s)
            break
            count += 1
            # spd = solver.solution_possible(s, end + 1)
            # if not spd["res"]:
            #     # print(spd["reason"], spd.get("detail",""))
            #     nomatchp.append(spd.get("detail", ""))
            #     reasons.append(spd["reason"])
            #     count2 += 1
            #     sol = s
            # else:
            #     print(s)
            # sol = s
            # print(s, spd["res"])
    print(Counter(reasons))
    print("count", count, count2)
    print(Counter(nomatchp))


def check_ao_data(s, end):
    import json

    with open("ayto_data.json", encoding="utf-8") as f:
        all_data = json.load(f)
    sn_key = "s4"
    season_data = all_data[sn_key]
    weeks = season_data["weeks"]
    print(season_data)
    for week in weeks[:end+1]:
        print("week number", week["number"])
        for event in week["events"]:
            if event["type"] == "box":
                a, b = event["pair"]
                if event["result"] == "yes" and not (a,b) in s:
                    return False, event
                if event["result"] == "no" and (a,b) in s:
                    return False, event
            elif event["type"] == "matching_night":
                lights = sum(1 for a, b in event["pairs"] if (a,b) in s)
                if lights != event["lights"]:
                    return False, event
            else: 
                print("other event", event)
    return True, None


if __name__ == "__main__":
    sn = "normalo2025"

    season: Season = get_season(sn)
    sol: Solution = season.solution  # type: ignore
    solver: Solver = get_solver(sn)
    # print(solver.no_match("Cecilia", "Felix", 6))
    end = 2
    from ayto.utils import time_it
    sols = solver.solve(end, True)
    plot_probs(sols)

    # prinr(probs)
    # amounts = []
    # for end in range(1, 10):
    #     sols = solver.solve(end, True)
    #     # sols = times = []
    #     print(len(sols))
    #     print(times)
    #     print(solver.times)
    #     amounts.append(len(sols))
    # print(amounts)

    # comparetoao(sols, end)
    exit() 

    s = Solution(
        frozenset(
            (
                (
                    ("Henna", "Kenneth"),
                    ("Juliette", "Barkin"),
                    ("Juliette", "Burim"),
                    ("Dorna", "Ken"),
                    ("Vanessa", "Max"),
                    ("Valeria", "Joel"),
                    ("Steffi", "Cris"),
                    ("Aurelia", "Sasa"),
                    ("Caro", "Deniz"),
                    ("Carina", "Pascal"),
                    ("Larissa", "Marwin"),
                )
            )
        )
    )
    print(s)
    res = check_ao_data(s, end)
    print(s.mm_left())
    print(res)

    # res = solver.solution_possible(s, end + 1)
    # print("res", res)
    # print("no_match", solver.no_match("Francesco", "Vanessa", end))
    # solspace = SolutionSpace(sols)
    # probs = solspace.get_pair_probs_dict()

    # df = pd.Series(probs).unstack(fill_value=0)
    # print(df)
    # analyze_solutions(sols)
    # plot_df(df)
