from collections import Counter
from memory_profiler import profile
import resource, sys
import timeit

from ayto import Solution, Solver, get_season
from ayto.utils import get_solver

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
    sn = "vip2022"
    season = get_season(sn)
    # print(season.get_matchboxes(7, False))
    print(season.get_black_out_nights(10, True))
    solver = get_solver(sn)


    end = 2
    icl = False # 4/14
    sols = solver.solve(end, icl)
    

    print(timeit.timeit(lambda: solver.solve(end, icl), number=1))

    sys.exit()
    def peak_mb() -> float:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024 if sys.platform == "darwin" else 1024)

    sols = get_solver("vip2025").solve(3, True)
    print(len(sols), f"{peak_mb():.0f} MB Spitze")
    print(sys.platform)


    
