import resource
import sys
from collections import Counter

from ayto import ALL_SEASONS, Solution, Solver, get_season, get_solver

# pyright: standard


def iterate_through(solver, start=3):
    amounts = []
    for end in range(start, 10):
        sols = solver.solve(end, True)
        amounts.append(len(sols))
    return amounts


def check_solution_possible(sols, solver: Solver, end:int, icl: bool):
    reasons = []
    for s in sols:
        p = solver.find_violation(s, end, icl)
        if p is not None:
            reasons.append(p)
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
    # uv run python -m scripts.debug_season
    sn = "vip2025"
    season = get_season(sn)
    # print(season.get_matchboxes(7, False))
    print(season.get_black_out_nights(10, True))
    solver: Solver = get_solver(sn)


    end = 2
    icl = True #  5_399_288 != np.int64(125_998)
    sols = solver.solve(end, icl)
    print(len(sols))
    # print(check_solution_possible(sols, solver, end, icl))

    import json
   

    # print(timeit.timeit(lambda: solver.solve(end, icl), number=1))

    sys.exit()
    for sn in ALL_SEASONS:
        season = get_season(sn)
        solver = get_solver(sn)
        d = {
            "match_model": season.match_model,
            "mm_person": season.mm_person,
            # "max_lights": season.max_lights,
            "mm_known_from_week": season.mm_known_from_week,
            "matchbox_reveals_dm": season.matchbox_reveals_dm,
        }
        if season.extra_left is not None:
            d["extra_left"] = season.extra_left
        if season.double_match_pair is not None:
            d["double_match_pair"] = season.double_match_pair
            d["double_match_pair_known"] = season.double_match_pair_known

        json_str = json.dumps(d, indent=4)
        with open(f"data/season_configs/{sn}.json","w") as f:
            f.write(json_str)

    def peak_mb() -> float:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024 if sys.platform == "darwin" else 1024)

    sols = get_solver("vip2025").solve(3, True)
    print(len(sols), f"{peak_mb():.0f} MB Spitze")
    print(sys.platform)



    
