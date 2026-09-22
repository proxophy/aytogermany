import pandas as pd
from collections import Counter

from ayto import Solver
from ayto.analysis import plot_probs, analyze_solutions
from ayto.models import Season, Solution, Pair
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
            # print(s)
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


if __name__ == "__main__":
    sn = "vip2025"

    tm = frozenset((str(i), str(i+100)) for i in range(10))
    tm_t = tuple(tm)

    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    season: Season = get_season(sn)
    sol: Solution = season.solution  # type: ignore
    solver: Solver = get_solver(sn)  # type: ignore
    end = 0
    sols = solver.solve(end, True, validate=False)
    profiler.disable()
    
    print(len(sols))
    import pstats
    p = pstats.Stats(profiler)
    p.sort_stats('cumulative').print_stats(10)
    # print(check_solution_possible(sols, solver, end))
    exit()

    pos_matches = {
            "A": ["1", "2"],
            "B": ["2", "3", "4"],
            "C": ["3", "4"],
        }
    
    lefts = ["A", "B", "C"]
    rights = ["1", "2", "3", "4"]
    from ayto.ayto import compute_matches

    psol = Solution(
        set(
            (('Nelly', 'Calvin O.'), ('Beverly', 'Nico'), ('Viki', 'Kevin'), ('Joanna', 'Rob'), ('Elli', 'Xander'), ('Henna', 'Oliver'), ('Hati', 'Sidar'), ('Sandra', 'Lennert'))
        )
    )


    print(sol.difference(psol))
    # print(sol.dict_rep())
    print(psol.issubset(sol))
    

    # sols = compute_matches(pos_matches, lefts, rights, None)
    # print(len(sols), sols)

    # psols = solver.generate_partial_solutions(end, True)
    # print(len(psols))
    # for p in psols:
    #     if (p.issubset(sol)):
    #         print(p)
    # for psol in psols:
    sols = solver.generate_complete_solutions(psol, end, True)
    print("sol in sols", sol in sols)
    
    # sols2 = solver.generate_complete_solutions2(psol, end, True)
    # print(len(sols), len(sols2))
    # if len(sols) != len(sols2):
    #     print(f"len(sols) != len(sols2) {len(sols)} {len(sols2)}")
    # for s in sols:
    #     if s not in sols2:
    #         print("not the same", s)

    exit()

    
