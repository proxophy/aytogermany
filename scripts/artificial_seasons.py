import itertools
import random
import string
from collections import Counter
from typing import Any

import tqdm

from ayto.codec import Codec
from ayto.season import (
    MATCH_MODELS,
    Matchboxes,
    Night,
    Pair,
    Season,
    matchboxes_from_list,
    mm_left,
    mm_right,
)
from ayto.utils import compare_solution_lists, get_solver_from_season

# pyright: standard
# , reportArgumentType=false


def simulate_season(
    n: int = 5,
    match_model: str = "double_match",
    dm_known: bool = True,
    matchbox_reveals_dm: bool = True,
) -> Season:

    def matchbox(i):
        mp = (random.choice(range(n)), random.choice(range(numrights)))
        while mp in mb_pairs:
            mp = (random.choice(range(n)), random.choice(range(numrights)))

        res = mp in solution
        matchboxes.append((i, mp, res, True))
        mb_pairs.append(mp)

        # double right match gets always revealed
        if res and match_model == "double_match_for_right" and mp[1] == 1:
            mllefts = iter({1, n} - {mp[0]})
            while (ml := next(mllefts, None)) is not None:
                np = (ml, 1)
                matchboxes.append((i, np, True, True))
                mb_pairs.append(np)

        if not matchbox_reveals_dm or not res:
            return
        # res = True
        # reveal double match (0,0), (0,n)
        if mp[0] == 0:
            mllefts = iter(
                ({0, n, n + 1} if match_model == "triple_match" else {0, n}) - {mp[1]}
            )
            while (ml := next(mllefts, None)) is not None:
                np = (0, ml)
                matchboxes.append((i, np, True, True))
                mb_pairs.append(np)
        # second multiple match (1,1), (1,n+1)
        if match_model == "two_double_matches" and mp[0] == 1:
            mllefts = iter({1, n + 1} - {mp[1]})
            while (ml := next(mllefts, None)) is not None:
                np = (1, ml)
                matchboxes.append((i, np, True, True))
                mb_pairs.append(np)

    def pop_dm(sol: set[Pair]):
        while (mml := mm_left(sol)) is not None:
            dmpairs = [(l, r) for (l, r) in sol if l == mml]
            sol.remove(dmpairs[0])

    def pop_right_dm(sol: set[Pair]):
        while (mmr := mm_right(sol)) is not None:
            dmpairs = [(l, r) for (l, r) in sol if r == mmr]
            sol.remove(dmpairs[0])

    lefts = tuple(string.ascii_uppercase)[
        : (n + 1 if match_model == "double_match_for_right" else n)
    ]
    numrights = (
        n + 2 if match_model in ("triple_match", "two_double_matches") else n + 1
    )
    rights = tuple(str(i) for i in range(1, numrights + 1))
    dm = "1"  # str(n + 1)
    solbase = {(l, r) for l, r in zip(range(n), range(n))}
    if match_model in ("double_match", "double_match_pair"):
        solution = frozenset(solbase | {(0, n)})
    elif match_model == "triple_match":
        solution = frozenset(solbase | {(0, n), (0, n + 1)})
    elif match_model == "two_double_matches":
        solution = frozenset(solbase | {(0, n), (1, n + 1)})
    elif match_model == "double_match_for_right":
        solution = frozenset(solbase | {(0, n), (n, 1)})

    else:
        raise NotImplementedError

    empty_season = Season(
        "empty",
        lefts,
        rights,
        (),
        Matchboxes(),
        match_model=match_model,
        mm_person=dm if dm_known else None,
        matchbox_reveals_dm=matchbox_reveals_dm,
        mm_known_from_week=0,
        double_match_pair_known=0,
        double_match_pair_names=(
            (str(1), str(n + 1)) if match_model == "double_match_pair" else None
        ),
        extra_left_name=lefts[-1] if match_model == "double_match_for_right" else None,
    )
    emptysolver = get_solver_from_season(empty_season)

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
        if match_model == "double_match_for_right":
            pop_right_dm(seated_pairs)

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
        match_model=match_model,
        mm_person=dm if dm_known else None,
        matchbox_reveals_dm=matchbox_reveals_dm,
        mm_known_from_week=0,
        double_match_pair_known=0,
        double_match_pair_names=(
            (str(1), str(n + 1)) if match_model == "double_match_pair" else None
        ),
        extra_left_name=lefts[-1] if match_model == "double_match_for_right" else None,
    )

    return season


def check_generated_season(season: Season, episode: int = 4, icl=True):
    errors: list[Any] = []

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
        if solver.find_violation(s, episode, icl) is not None:
            errors.append(
                f"solution in sols_by_solve not possible {solver.find_violation(s, episode, icl)}"
            )

    if len(sols_filtered) != len(sols_by_solve):
        errors.append(
            f"len(sols_filtered) = {len(sols_filtered)} != {len(sols_by_solve)} = len(sols_by_solve)"
        )

    if len(sols_filtered) == 0:
        return Counter(errors)
    if len(sols_filtered) == len(sols_by_solve) == 0:
        errors.append("len(sols_filtered) == len(sols_by_solve) == 0")

    if not compare_solution_lists(sols_filtered, sols_by_solve):
        errors.append("sols_filtered != sols_by_solve")

    bitsolver = get_solver_from_season(season, True)
    ot_sols = bitsolver.solve(episode, icl)
    sols_by_solve = [codec.encode(s) for s in sols_by_solve]

    if len(sols_by_solve) != len(ot_sols):
        errors.append("len(sols_by_solve) != len(ot_sols)")

    if sorted(sols_by_solve) != sorted(ot_sols):
        errors.append("sorted(sols_by_solve) != sorted(ot_sols)")
    return Counter(errors)


def stress_test(n: int = 5, model="double_match", matchbox_reveals_dm=False):
    print(f"match_model = {model} with matchbox_reveals_dm = {matchbox_reveals_dm}")
    done = False
    report = []
    for i in tqdm.trange(250):
        season = simulate_season(
            5, match_model=model, matchbox_reveals_dm=matchbox_reveals_dm
        )
        for ep, icl in itertools.product(range(4), [True, False]):

            try:
                errors = check_generated_season(season, episode=ep, icl=icl)
                if len(errors) > 0:
                    done = True
                    report.append((ep, icl, errors, season))
                    break
            except Exception as e:
                done = True
                report.append((ep, icl, e, season))
                break
        if done:
            print(
                f"match_model = {model} with matchbox_reveals_dm = {matchbox_reveals_dm} has error"
            )
            break
    return report


if __name__ == "__main__":
    # uv run python -m scripts.artificial_seasons
    # random.seed(42)
    ep = 5
    icl = True
    model = "two_double_matches"
    errors = stress_test(model=model, matchbox_reveals_dm=False)
    print(errors)

    for model in MATCH_MODELS:
        for matchbox_reveals_dm in [True, False]:
            if not matchbox_reveals_dm and model not in [
                "double_match",
                "double_match_for_right",
            ]:
                continue
            stress_test(model=model, matchbox_reveals_dm=matchbox_reveals_dm)
