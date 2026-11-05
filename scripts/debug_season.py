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


def check_solution_possible(sols, solver: Solver, end: int, icl: bool):
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


def peak_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024 if sys.platform == "darwin" else 1024)


def memory_test():
    import timeit

    from pympler import asizeof

    sn = "vip2025"
    solver: Solver = get_solver(sn)
    end = 3
    icl = False

    sols = solver.solve(end, icl)
    print(len(sols))
    print(f"{timeit.timeit(lambda: print(len(solver.solve(end, icl))), number=1):.2f}")
    print(f"Total: {asizeof.asizeof(sols) / 1024**3:.3f} GB")
    print(f"Number of solutions: {len(sols):,}")
    print(f"Average per solution: " f"{asizeof.asizeof(sols) / len(sols):.1f} bytes")

    # sols = get_solver("vip2025").solve(3, True)
    print(f"{peak_mb():.0f} MB Spitze")


if __name__ == "__main__":
    # uv run python -m scripts.debug_season
    sn = "vip2025"
    season = get_season(sn)
    solver: Solver = get_solver(sn)
    import timeit
    import pandas as pd

    end = 2
    icl = True



    # sol = season.solution
    # sols = solver.solve(end, icl)
    # print(len(sols))
    # print(timeit.timeit(lambda solver=solver, end=end: len(solver.solve(end, icl)), number=3))

    # sys.exit()
    data = []
    for sn in ALL_SEASONS:
        solver = get_solver(sn)
        td = {}
        for end in range(1, 3):
            x = timeit.timeit(lambda solver=solver, end=end: len(solver.solve(end, icl)), number=3)
            td[end] = x
        data.append(td)
    df = pd.DataFrame(data, index=ALL_SEASONS)
    # df.to_csv("analytics/time")
    print(df)

    # memory_test()
    # from pympler import asizeof
    # solution = next(iter(sols))

    # print("Solution:", asizeof.asizeof(solution))
    # print("Pair:", asizeof.asizeof(next(iter(solution))))
    # print("tuple[str,str]", asizeof.asizesof(("Laurenz", "Joena")))
    # print("Pair type:", type(next(iter(solution))))
