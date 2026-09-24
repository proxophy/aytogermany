import pandas as pd
from collections import Counter

from ayto import Solver, allseasons
from ayto.analysis import plot_probs, analyze_solutions, matching_night_probs
from ayto.models import Season, Solution, Pair, get_candidates
from ayto.utils import get_season, get_solver
from ayto.aytovip25 import VIP2025Solver


def appendsol(sn: str, tsol: set[tuple[str, str]]):
    plefts, prights = zip(*tsol)
    dfsol = pd.DataFrame(data={"left": plefts, "right": prights})
    dfsol = dfsol.sort_values(by=["left", "right"]).reset_index(drop=True)
    print(dfsol)
    with pd.ExcelWriter(
        f"data/{sn}.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        dfsol.to_excel(writer, sheet_name="Solution", index=False)


def comparetoao(solver, sols, end):
    with open("aodata.txt", "r") as f:
        lines = map(eval, f.readlines())

    def parse_sol(ld):
        pairs = set()
        for l, rs in ld.items():
            for r in rs:
                pairs.add((l, r))
        return Solution(set(pairs))

    solsao = list(map(parse_sol, lines))

    print("lengths", len(solsao), len(sols))
    count = count2 = 0
    reasons = []
    for s in solsao:
        spd = solver.solution_possible(s, end, True)
        if not spd["res"]:
            count += 1
            reasons.append(spd["reason"])
        elif s not in sols:
            count2 += 1
            print(s)
            # break

    print(
        f"solution in solsao not possible: {count}; solutions in solsao not in sols {count2}"
    )
    print(len(reasons), Counter(reasons))

    count = count2 = 0
    nomatchp = []
    reasons = []
    for s in sols:
        if s not in solsao:
            spd = solver.solution_possible(s, end, False)
            count += 1
            # print(s)
            if not spd["res"]:
                reasons.append(spd["reason"])
                # print(spd)
                count2 += 1
    print(Counter(reasons))
    print(f"sols not in solsao: {count}; {count2}")
    # print(Counter(nomatchp))


def check_ao_data(sols: list[Solution], end, sn):
    import json

    with open("ayto_data.json", encoding="utf-8") as f:
        all_data = json.load(f)
    season_data = all_data[sn]
    weeks = season_data["weeks"]
    # print("end", end)
    for week in weeks[: end + 1]:
        # print("week number", week["number"])
        for event in week["events"]:
            if event["type"] == "box":
                a, b = event["pair"]
                if event["result"] == "yes" and not (a, b) in sols:
                    return False, event
                if event["result"] == "no" and (a, b) in sols:
                    return False, event
            elif event["type"] == "matching_night":
                lights = sum(1 for a, b in event["pairs"] if (a, b) in sols)
                if lights != event["lights"]:
                    return False, event
            # else:
            #     print("other event", event)
    return True, None


def iterate_through(solver, start=3):
    amounts = []
    for end in range(start, 10):
        sols = solver.solve(end, True)
        amounts.append(len(sols))
    return amounts


def check_solution_possible(sols, solver: Solver, end):
    reasons = []
    for s in sols:
        p = solver.solution_possible(s, end, True)
        if not p["res"]:
            reasons.append(p["reason"])
            # print(s, p["reason"], p.get("detail", ""))
    return Counter(reasons)


def findpsol(sol, solver: Solver, end):
    psols = solver.generate_partial_solutions(end, True)
    psol = Solution()
    for p in psols:
        if sol in solver.generate_complete_solutions(p, end, True):
            return p
    return psol


def toy_example():
    from ayto.models import Matchboxes, Night

    toy_nights = (
        Night(
            set(
                (
                    ("A", "1"),
                    ("B", "3"),
                    ("C", "4"),
                    ("D", "5"),
                    ("E", "2"),
                )
            ),
            1,
        ),
        Night(
            set(
                (
                    ("A", "5"),
                    ("B", "2"),
                    ("C", "3"),
                    ("D", "4"),
                    ("E", "1"),
                )
            ),
            3,
        ),
        Night(
            set(
                (
                    ("A", "5"),
                    ("B", "2"),
                    ("C", "3"),
                    ("D", "4"),
                    ("E", "6"),
                )
            ),
            4,
        ),
    )
    toy_matchboxes = Matchboxes((0,), ((("B", "3")),), (False,))
    toy_lefts = ("A", "B", "C", "D", "E")
    toy_rights = ("1", "2", "3", "4", "5", "6")
    toy_season = Season(
        "toy", toy_lefts, toy_rights, toy_nights, toy_matchboxes, mm="6", num_matches=6
    )
    toy_seasonwo = Season(
        "toy", toy_lefts, toy_rights, toy_nights, Matchboxes(), mm="6", num_matches=6
    )
    toy_solver = Solver(toy_season)

    sol = frozenset({("E", "2"), ("E", "6"), ("A", "5"), ("D", "4"), ("C", "3")})
    print(get_candidates(sol))
    # solver.generate_complete_solutions(sol, 5, True)
    sols = toy_solver.solve(5)
    # sols_as_df(diff).sort_values(by= ["1", "2", "3", "4", "5", "6"]) [toy_rights].to_csv("solsdf.csv")


if __name__ == "__main__":
    sn = "vip2025"

    import cProfile

    profiler = cProfile.Profile()
    profiler.enable()

    season: Season = get_season(sn)
    sol: Solution = season.solution  # type: ignore
    solver: Solver = get_solver(sn)  # type: ignore
    end = 8
    sols = solver.solve(end)
    print(len(sols)) 

    # comparetoao(solver, sols, end)
    # analyze_solutions(sols)
    # print(check_solution_possible(sols, solver, end))
    # plot_probs(sols)

    sol = frozenset({('Henna', 'Leandro'), ('Sandra', 'Lennert'), ('Viki', 'Jimi'), ('Henna', 'Oliver'), ('Hati', 'Jonny'), ('Joanna', 'Nico'), ('Nelly', 'Calvin O.'), ('Ariel', 'Rob'), ('Viki', 'Kevin'), ('Beverly', 'Calvin S.'), ('Antonia', 'Sidar'), ('Elli', 'Xander')})
    psol = frozenset({('Viki', 'Kevin'), ('Elli', 'Xander'), ('Hati', 'Jonny'), ('Henna', 'Oliver'), ('Beverly', 'Calvin S.'), ('Nelly', 'Calvin O.'), ('Sandra', 'Lennert'), ('Henna', 'Leandro')})
    # for psol in solver.generate_partial_solutions(end, True):
    #     if psol <= sol:
    #         print(psol)

    from ayto.utils import dict_rep, has_two_dms
    print(dict_rep(psol), has_two_dms(psol), solver.two_dms)
    # sols = solver.generate_complete_solutions(psol, end, True)
    # print("sol ins sols", sol in psol)

    profiler.disable()
    import pstats

    p = pstats.Stats(profiler)
    # p.sort_stats("cumulative").print_stats(10)
    # toy_example()
    exit()

    for sn in allseasons:
        season = get_season(sn)
        solver = get_solver(sn)
        print(sn, season.get_black_out_nights())
        # print(matching_night_probs(solver, 7))
