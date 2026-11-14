import resource
import sys
from collections import Counter

from ayto import (
    ALL_SEASONS,
    TupleSolver,
    get_season,
    get_solver,
)
from ayto.base import Solver
from ayto.bitmask_solver import BitMaskSolver
from ayto.season import MATCH_MODELS
from ayto.two_double_matches_solver import TwoDoubleMatchesBitMaskSolver
from ayto.utils import get_solver_type
from tests.test_solver_logic import (
    model_to_toy,
)

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


def findpsol(sol, solver: Solver, end, icl):
    psols = solver.generate_partial_solutions(end, icl)
    for p in psols:
        if sol in solver.generate_complete_solutions(p, end, icl):
            return p
    return None


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


def check_all_solutions(match_model: str):
    season = model_to_toy[match_model]
    empty = season.as_empty()
    empty_tsolver = get_solver_type(empty, False)(empty)
    empty_bmsolver = get_solver_type(empty, True)(empty)
    tsolver = get_solver_type(season)(season)
    bmsolver = get_solver_type(season, True)(season)
    codec = bmsolver.codec

    end = 0
    icl = False

    sols = empty_tsolver.generate_complete_solutions(frozenset(), end, True)
    sols = set(sols)
    msols = empty_bmsolver.generate_complete_solutions(0, end, True)
    msols = set(msols)
    print(f"For match model {match_model} {len(sols)} (t) {len(msols)} (bm)")
    print(tsolver.__class__, bmsolver.__class__)
    count = 0

    reasons_t: list[str | None] = []
    reasons_b: list[str | None] = []
    for se in msols:
        s = codec.decode(se)
        # se = codec.encode(s)
        if (tv := tsolver.find_violation(s, end, icl)) != (
            bv := bmsolver.find_violation(se, end, icl)
        ):
            count += 1
            reasons_t.append(tv)
            reasons_b.append(bv)
    return count, Counter(reasons_t), Counter(reasons_b)


if __name__ == "__main__":
    # uv run python -m scripts.debug_season
    sn = "vip2022"
    season = get_season(sn)
    bmsolver = get_solver(sn, True)
    codec = bmsolver.codec
    tsolver = get_solver(sn, False)
    end = 7
    icl = False
    # print(len(bsols := bmsolver.solve(end, icl)))
    # print(len(tsols := tsolver.solve(end, icl)))


    # print(check_all_solutions("double_match_for_right"))

    for model in MATCH_MODELS:
        print(check_all_solutions(model))
        print()

    # memory_test()
    sys.exit()
    import timeit

    import pandas as pd

    data = []
    for sn in ALL_SEASONS:
        solver = get_solver(sn, True)
        td = {}
        for end in range(10):
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
