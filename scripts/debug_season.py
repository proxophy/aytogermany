import resource
import sys
from collections import Counter

from ayto import ALL_SEASONS, Solution, TupleSolver, get_season, get_solver
from ayto.bitmask_solver import BitMaskSolver
from ayto.two_double_matches_solver import TwoDoubleMatchesBitMaskSolver

# pyright: standard


def iterate_through(solver, start=3):
    amounts = []
    for end in range(start, 10):
        sols = solver.solve(end, True)
        amounts.append(len(sols))
    return amounts


def check_solution_possible(
    sols, solver: TupleSolver | BitMaskSolver, end: int, icl: bool
):
    reasons = []
    for s in sols:
        p = solver.find_violation(s, end, icl)
        if p is not None:
            reasons.append(p)
            # print(s, p["reason"], p.get("detail", ""))
    return Counter(reasons)


def findpsol(sol, solver: TupleSolver, end):
    psols = solver.generate_partial_solutions(end, True)
    psol = Solution()
    for p in psols:
        if sol in solver.generate_complete_solutions(p, end, True):
            return p
    return psol


def peak_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024 if sys.platform == "darwin" else 1024)


def memory_test():

    sn = "vip2025"
    season = get_season(sn)
    solver = TwoDoubleMatchesBitMaskSolver(season)
    end = 2
    icl = True

    start = peak_mb()

    sols = solver.solve(end, icl)
    print(f"{peak_mb()} nach solve")
    # print(f"{timeit.timeit(lambda: print(len(solver.solve(end, icl))), number=1):.2f} s")
    # print(f"Total: {asizeof.asizeof(sols) / 1024**3:.3f} GB")
    # print(f"Number of solutions: {len(sols):,}")
    # print(f"Average per solution: " f"{asizeof.asizeof(sols) / len(sols):.1f} bytes")

    print(
        f"Start {start:.0f} MB, danach {peak_mb():.0f} MB, "
        f"+{peak_mb() - start:.0f} MB für {len(sols):,} Lösungen"
    )


if __name__ == "__main__":
    # uv run python -m scripts.debug_season
    sn = "vip2023"
    season = get_season(sn)
    solver2 = get_solver(sn, True)
    solver1 = get_solver(sn)
    solution = season.solution
    sol_enc = solver2.codec.encode(solution)

    # print(type(solver2))
    import timeit

    import pandas as pd

    end = 9
    icl = True

    sols1 = solver1.solve(end, icl)
    sols2 = solver2.solve(end, icl)
    print(len(sols2), len(set(sols2)))

    # memory_test()
    sys.exit()

    data = []
    for sn in ALL_SEASONS:
        solver = get_solver(sn)
        td = {}
        for end in range(2, 10):
            x = timeit.timeit(
                lambda solver=solver, end=end: len(solver.solve(end, True)), number=3
            )
            td[end] = x
        data.append(td)
    df = pd.DataFrame(data, index=ALL_SEASONS)
    # df.to_csv("analytics/time")
    print(df)

    # from pympler import asizeof
    # solution = next(iter(sols))
