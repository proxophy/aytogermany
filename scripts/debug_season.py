import resource
import sys
from collections import Counter

from ayto import Solution, Solver, get_season, get_solver

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


if __name__ == "__main__":
    # uv run python -m scripts.debug_season
    sn = "vip2025"
    season = get_season(sn)
    solver: Solver = get_solver(sn)


    end = 3
    icl = False
    psols = solver.generate_partial_solutions(end, icl)
    print(len(psols))
    sols = solver.solve(end, icl)
    print(len(sols))
    from pympler import asizeof

    print(f"Total: {asizeof.asizeof(sols) / 1024**3:.3f} GB")
    print(f"Number of solutions: {len(sols):,}")
    print(f"Average per solution: " f"{asizeof.asizeof(sols) / len(sols):.1f} bytes")

    import timeit

    lens = []

    # sys.exit()


    def peak_mb() -> float:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024 if sys.platform == "darwin" else 1024)

    print(f"{timeit.timeit(lambda: print(len(solver.solve(end, icl))), number=1):.2f}")

    # sols = get_solver("vip2025").solve(3, True)
    print(f"{peak_mb():.0f} MB Spitze")
