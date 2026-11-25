import resource
import sys
from collections import Counter

from ayto import (
    ALL_SEASONS,
    get_season,
    get_solver,
)
from ayto.base import SolutionType, Solver
from ayto.two_double_matches_solver import TwoDoubleMatchesBitMaskSolver
from ayto.utils import get_solver_from_season
from tests.fixtures import (
    ALL_TOY_SEASONS,
)

# pyright: standard


def iterate_through(solver, start=3):
    amounts = []
    for ep in range(start, 10):
        sols = solver.solve(ep, True)
        amounts.append(len(sols))
    return amounts


def check_solution_possible(sols, solver: Solver, ep: int, icl: bool):
    reasons = []
    for s in sols:
        p = solver.find_violation(s, ep, icl)
        if p is not None:
            reasons.append(p)
            # print(s, p["reason"], p.get("detail", ""))
    return Counter(reasons)


def findpsol[S : SolutionType,T](sol: S, solver: Solver[S,T], episode: int, icl: bool):
    psols = solver.generate_partial_solutions(episode, icl)
    for p in psols:
        if sol in solver.generate_complete_solutions(p, episode, icl):
            return p
    return None


def peak_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024 if sys.platform == "darwin" else 1024)


def memory_test():

    sn = "vip2025"
    season = get_season(sn)
    solver = TwoDoubleMatchesBitMaskSolver(season)
    ep = 2
    icl = True

    start = peak_mb()

    sols = solver.solve(ep, icl)
    print(f"{peak_mb()} nach solve")
    # print(f"{timeit.timeit(lambda: print(len(solver.solve(ep, icl))), number=1):.2f} s")
    # print(f"Total: {asizeof.asizeof(sols) / 1024**3:.3f} GB")
    # print(f"Number of solutions: {len(sols):,}")
    # print(f"Average per solution: " f"{asizeof.asizeof(sols) / len(sols):.1f} bytes")

    print(
        f"Start {start:.0f} MB, danach {peak_mb():.0f} MB, "
        f"+{peak_mb() - start:.0f} MB für {len(sols):,} Lösungen"
    )


def check_all_solutions(match_model: str):
    season = ALL_TOY_SEASONS[match_model]
    empty = season.as_empty()
    empty_tsolver = get_solver_from_season(empty)
    empty_bmsolver = get_solver_from_season(empty, True)
    tsolver = get_solver_from_season(season)
    bmsolver = get_solver_from_season(season, True)
    codec = bmsolver.codec

    ep = 0
    icl = False

    sols = empty_tsolver.generate_complete_solutions(frozenset(), ep, True)
    sols = set(sols)
    msols = empty_bmsolver.generate_complete_solutions(0, ep, True)
    msols = set(msols)
    print(f"For match model {match_model} {len(sols)} (t) {len(msols)} (bm)")
    print(tsolver.__class__, bmsolver.__class__)
    count = 0

    reasons_t: list[str | None] = []
    reasons_b: list[str | None] = []
    for se in msols:
        s = codec.decode(se)
        # se = codec.encode(s)
        if (tv := tsolver.find_violation(s, ep, icl)) != (
            bv := bmsolver.find_violation(se, ep, icl)
        ):
            count += 1
            reasons_t.append(tv)
            reasons_b.append(bv)
    return count, Counter(reasons_t), Counter(reasons_b)


if __name__ == "__main__":
    # uv run python -m scripts.debug_season
    sn = "vip2026"
    season = get_season(sn)
    # season = toy_season_with_dm_for_right
    # bmsolver = get_solver(sn, True)
    
    # solver_type = get_solver_type(season)
    bitsolver = get_solver_from_season(season, True)
    tupsolver = get_solver_from_season(season, False)
    codec = bitsolver.codec

    empty = season.as_empty()
    solver_for_empty = get_solver_from_season(empty, True)

    ep = 5
    icl = True
    print(tupsolver.__class__)
    print(bitsolver.__class__)
    # tsols = tupsolver.solve(ep, icl)
    # bsols = bitsolver.solve(ep, icl)
    # print(len(tsols), len(bsols))

    # for ep in range(1, 10):
    #     psols = bitsolver.generate_partial_solutions(ep, icl)
    #     lens = []
    #     for ps in psols:
    #         sols = bitsolver.generate_complete_solutions(ps, ep, icl)
    #         lens.append(len(sols))
    #     print(
    #         f"ep={ep}: {len(psols)} psols, {max(lens)} max, {mean(lens):0.1f} mean, {sum(lens)} sum"
    #     )

    # plot_probs(tsols)

    # for model in MATCH_MODELS:
    #     print(check_all_solutions(model))
    #     print()

    # memory_test()
    sys.exit()
    import timeit

    import pandas as pd

    data = []
    for sn in ALL_SEASONS:
        bitsolver = get_solver(sn, True)
        td = {}
        for ep in range(10):
            x = timeit.timeit(
                lambda solver=bitsolver, ep=ep: len(solver.solve(ep, True)), number=3
            )
            td[ep] = x
        data.append(td)
    df = pd.DataFrame(data, index=ALL_SEASONS)
    # df.to_csv("analytics/time")
    print(df)

    # from pympler import asizeof
    # solution = next(iter(sols))
