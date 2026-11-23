import random
import string

from ayto.codec import Codec
from ayto.season import (
    Matchboxes,
    Night,
    Pair,
    Season,
    matchboxes_from_list,
    mm_left,
)
from ayto.tuple_solver import TupleSolver
from ayto.utils import compare_solution_lists, get_solver_from_season

# pyright: standard
# , reportArgumentType=false


def simulate_season(n: int = 5, with_dm: bool = True) -> Season:

    def matchbox(i):
        mp = (random.choice(range(n)), random.choice(range(n + 1)))
        while mp in mb_pairs:
            mp = (random.choice(range(n)), random.choice(range(n + 1)))

        res = mp in solution
        matchboxes.append((i, mp, res, True))
        mb_pairs.append(mp)

        # reveal double match (0,0), (0,5)
        if res and mp[0] == 0:
            mr = next(iter({0, n} - {mp[1]}))
            np = (0, mr)
            matchboxes.append((i, np, True, True))
            mb_pairs.append(np)

    def pop_dm(sol):
        mml = mm_left(sol)
        dmpairs = [(l, r) for (l, r) in sol if l == mml]
        sol.remove(dmpairs[0])

    lefts = tuple(string.ascii_uppercase)[:n]
    rights = tuple(str(i) for i in range(1, n + 2))
    dm = str(n + 1)
    solution = frozenset({(l, r) for l, r in zip(range(n), range(n))} | {(0, n)})

    empty_season = Season(
        "empty",
        lefts,
        rights,
        (),
        Matchboxes(),
        mm_person=dm if with_dm else None,
        mm_known_from_week=0,
    )
    emptysolver = TupleSolver(empty_season)

    nights = []
    matchboxes = []
    mb_pairs: list[Pair] = []

    seated_pairs = set()
    clights = 0

    i = 0
    while i < n:
        matchbox(i)

        # choose new pairs
        pairs = random.sample(list(seated_pairs), clights)
        pairs = set()
        gsols = emptysolver.generate_complete_solutions(frozenset(pairs), n, True)
        seated_pairs = set(
            random.choice(gsols)
        )  # choose a random valid pair combination
        pop_dm(seated_pairs)

        clights = len(seated_pairs & solution)  # determine lights
        nights.append(Night(seated_pairs, clights))

        i += 1

    season = Season(
        "generated",
        lefts,
        rights,
        tuple(nights),
        matchboxes_from_list(matchboxes),
        solution=solution,
        mm_person=dm if with_dm else None,
        matchbox_reveals_dm=True,
        mm_known_from_week=0,
    )

    return season


def check_generated_season(season: Season, episode: int = 4, icl=True) -> list[str]:
    errors = []

    solver = get_solver_from_season(season)
    empty_solver = solver.as_empty()
    codec = Codec(len(season.lefts), len(season.rights))
    assert solver.__class__ == empty_solver.__class__

    sols_filtered = [
        s
        for s in empty_solver.solve(episode)
        if solver.solution_possible(s, episode, icl)
    ]
    sols_by_solve = solver.solve(episode, icl)

    solution = season.solution
    assert solution

    for l, r in solution:
        if solver.no_match(l, r, episode, icl):
            errors.append("no match in solution")
    if solver.find_violation(solver.from_pairs(solution), episode, icl) is not None:
        errors.append("solution not possible")

    for s in sols_by_solve:
        if not solver.solution_possible(s, episode, True):
            errors.append(
                f"solution in sols_by_solve not possible {solver.find_violation(s, episode, icl)}"
            )

    if len(sols_filtered) != len(sols_by_solve):
        errors.append(
            f"len(sols) = {len(sols_filtered)} != {len(sols_by_solve)} = len(sols_by_solve)"
        )

    if len(sols_filtered) == 0:
        return errors
    if len(sols_filtered) == len(sols_by_solve) == 0:
        errors.append("len(sols_filtered) == len(sols_by_solve) == 0")

    if not compare_solution_lists(sols_filtered, sols_by_solve):
        errors.append("sols_filtered != sols_by_solve")

    other_type_solver = get_solver_from_season(season, True)
    ot_sols = other_type_solver.solve(episode, icl)
    sols_by_solve = [codec.encode(s) for s in sols_by_solve]

    if len(sols_by_solve) != len(ot_sols):
        errors.append("len(sols_by_solve) != len(ot_sols)")

    if sorted(sols_by_solve) != sorted(ot_sols):
        errors.append("sorted(sols_by_solve) != sorted(ot_sols)")
    return errors


if __name__ == "__main__":
    # uv run python -m scripts.artificial_seasons

    # random.seed(42)
    ep = 1
    icl = True
    season = simulate_season(5, False)
    solution = season.solution
    solver = get_solver_from_season(season)
    print(solver.solution_possible(solution, ep, icl))

    # for l, r in solution:
    #     if nm := solver.no_match_reason(l, r, ep, icl):
    #         print(l, r, nm)

    # print(check_generated_season(season, ep, icl))

    # for i in tqdm.trange(100):
    #     season = simulate_season(5, False)
    #     for ep in range(4):
    #         for icl in [True, False]:
    #             if len(errors := check_generated_season(season, episode=ep)) > 0:
    #                 print(ep, icl, errors, season)
    #                 sys.exit()
