import itertools
import timeit

from ayto import *
from ayto.bitmask_solver import BitMaskSolver, bitmask_product_without_reps
from ayto.codec import Codec, Mask
from ayto.solver import product_without_reps


def time_comparison_product_without_reps():
    import timeit

    masks = [
        0b0000111111,
        0b0111111000,
        0b1111110000,
        0b1111000011,
    ]

    print(
        timeit.timeit(
            lambda: sum(1 for _ in bitmask_product_without_reps(masks)), number=1000
        )
    )

    lists = [[i for i in range(mask.bit_length()) if mask & (1 << i)] for mask in masks]
    print(
        timeit.timeit(
            lambda: sum(1 for _ in product_without_reps(lists)),
            number=1000,
        )
    )


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
    sn = "normalo2023"
    season = get_season(sn)
    # solver = get_solver(sn)
    # season, _ = simulate_season(4)
    # season = toy_example()

    solver = Solver(season)
    solution = season.solution
    # print(len(solver.solve(5,True)))

    codec = Codec(len(season.lefts), len(season.rights))
    bmsolver = BitMaskSolver(season)

    t1 = frozenset({(3, 3), (2, 2), (0, 0), (1, 1), (0, 4)})
    # t1 = season.nights[4].pairs
    # t1m = codec.encode(t1)

    episode = 1

    print("default solver")
    # print(len(bmsolver.combine_possible_solutions(pairs[0], pairs[1], 1, True)))
    print(
        "len(psols1):", len(psols1 := solver.generate_partial_solutions(episode, True))
    )
    t1 = psols1[0]

    sols1 = solver.solve(episode, True)

    print(t1)

    print(f"{timeit.timeit(lambda:  solver.solve(episode, True), number=3):0.5f}")

    print("bm solver")

    print(
        "len(psols1)", len(psols2 := bmsolver.generate_partial_solutions(episode, True))
    )
    t1m = codec.encode(t1)
    knowledgeb = bmsolver.knowledge_cache[(episode, True)]

    sols2: list[Mask] = bmsolver.solve(episode, True)

    print("len(sols1)", len(sols1))
    print("len(sols2)", len(sols2))

    print(f"{timeit.timeit(lambda: bmsolver.solve(episode, True), number=3):0.5f}")

    # for sol in sols1:
    #     x = codec.encode(sol)
    #     if x not in sols2:
    #         print("not in:", x)
    # x = sols2[0]
    # t1 = codec.decode(x)
    # d = dict_rep(solution)
    # print(sols2)
    # for k,v in dict_rep(t1).items():
    #     print(k,v, "/", d[k])

    # print(psols2[0].bit_count())

    # sols = bmsolver.generate_complete_solutions(164794897195567826914958328729616, episode, True)
    # print("sols", sols)

    # print("eq", {codec.encode(s) for s in psols1} == set(psols2))
