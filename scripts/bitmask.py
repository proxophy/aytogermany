import itertools
import sys  # pyright: ignore[reportUnusedImport]  # noqa: F401
import timeit

from ayto import *
from ayto.codec import Codec, Mask


def compare_no_matches():
    end = len(season.lefts)

    all_pairs = list(itertools.product(season.lefts, season.rights))

    no_matches = [
        (l, r) for (l, r) in all_pairs if solver.no_match(l, r, end, True)
    ]  # 66
    print("default solver", len(no_matches))
    st = timeit.timeit(
        lambda: [(l, r) for (l, r) in all_pairs if solver.no_match(l, r, end, True)],
        number=100,
    )
    print(f"{st:0.5f}")

    print(
        "bmsolver",
        len(
            no_matches_bm := [
                (l, r) for (l, r) in all_pairs if bmsolver.no_match(l, r, end, True)
            ]
        ),
    )
    bst = timeit.timeit(
        lambda: [(l, r) for (l, r) in all_pairs if bmsolver.no_match(l, r, end, True)],
        number=100,
    )
    print(f"{bst:0.5f}")

    print("eq", no_matches == no_matches_bm)


if __name__ == "__main__":
    # uv run python -m scripts.bitmask
    # uv run kernprof --l -v -m scripts.bitmask

    sn = "vip2026"
    season = get_season(sn)

    solver = get_solver(sn)
    solution = season.solution

    codec = Codec(len(season.lefts), len(season.rights))
    bmsolver = get_solver(sn, True)
    sol_enc = codec.encode(solution)

    episode = 1
    icl = True

    print("default solver", type(solver))
    print(
        "len(psols1):", len(psols1 := solver.generate_partial_solutions(episode, icl))
    )
    # t1 = psols1[0]
    # sols1: list[Solution] = solver.solve(episode, icl)
    # print("len(sols1)", len(sols1))
    # print(f"{timeit.timeit(lambda:  solver.solve(episode, include_night=icl), number=3):0.5f}")
    # sols1_enc = [codec.encode(s) for s in sols1]

    print("bm solver", type(bmsolver))
    # assert season.extra_left
    print(bmsolver.find_violation(sol_enc, episode, icl))
    # print(check_solution_possible(sols1_enc, bmsolver, episode, icl))
    # print(bin(codec.multi_rights(sol_enc)))
    # print(season.rights_names)

    print(
        "len(psols1)", len(psols2 := bmsolver.generate_partial_solutions(episode, icl))
    )
    # print("solution possible", bmsolver.find_violation(sol_enc, episode, icl))
    sols2: list[Mask] = bmsolver.solve(episode, icl)
    print("len(sols2)", len(sols2))

    # print(f"{timeit.timeit(lambda: bmsolver.solve(episode, icl), number=3):0.5f}")
