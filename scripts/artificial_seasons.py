import random
import string

from ayto import Matchboxes, Night, Season, Solution, TupleSolver
from ayto.season import matchboxes_from_list, mm_left

# pyright: standard


def toy_example():

    toy_nights = (
        Night(
            {
                (0, 0),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 1),
            },
            1,
        ),
        Night(
            {
                (0, 4),
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 0),
            },
            3,
        ),
        Night(
            {
                (0, 4),
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 5),
            },
            4,
        ),
    )
    toy_matchboxes = Matchboxes((0,), (((1, 2)),), (False,), (True,))
    toy_lefts = ("A", "B", "C", "D", "E")
    toy_rights = ("0", "1", "2", "3", "4", "5")
    toy_season = Season(
        "toy", toy_lefts, toy_rights, toy_nights, toy_matchboxes, mm_person="5"
    )
    solver = TupleSolver(toy_season)

    sols = solver.solve(5)
    return toy_season


def simulate_season(n: int = 5, with_dm: bool = True) -> tuple[Season, list[Solution]]:

    def matchbox(sols, i):
        mp = (random.choice(range(n)), random.choice(range(n + 1)))
        if mp in solution:
            sols = [s for s in sols if mp in s]
            known_pms.append(mp)
        else:
            sols = [s for s in sols if mp not in s]

        res = mp in solution
        mbs.append((i, mp, res, True))
        # reveal double match
        if res and mp[0] == 0:
            mr = next(iter({0, n} - {mp[1]}))
            np = (0, mr)
            mbs.append((i, np, True, True))
            sols = [s for s in sols if np in s]
            known_pms.append(np)
            # print("double match", mp, mr)
        elif res:
            # double not possible with this l
            sols = [s for s in sols if (mp[0], n) not in s]
            for l in lefts:
                if l == mp[0]:
                    continue
                # filter out solutions that have mp[1] with another partner
                sols = [s for s in sols if (l, mp[1]) not in s]
            for r in rights:
                if r == mp[1]:
                    continue
                # filter out solutions that have mp[0] with another partner
                sols = [s for s in sols if (mp[0], r) not in s]
        return sols

    def pop_dm(sol):
        mml = mm_left(sol)
        dmpairs = [(l, r) for (l, r) in sol if l == mml]
        sol.remove(dmpairs[0])

    lefts = tuple(string.ascii_uppercase)[:n]
    rights = tuple(str(i) for i in range(1, n + 2))
    dm = str(n + 1)
    solution = frozenset({(l, r) for l, r in zip(range(n), range(n))} | {(0, n)})

    empty_season = Season(
        "toy",
        lefts,
        rights,
        (),
        Matchboxes(),
        mm_person=dm if with_dm else None,
        mm_known_from_week=0,
    )
    emptysolver = TupleSolver(empty_season)

    sols = emptysolver.generate_complete_solutions(Solution(), n, True)
    sols = list(set(sols))

    nights = []
    mbs = []
    known_pms = []  # track perfect matches

    # print("before:", len(sols))
    seated_pairs = set()
    clights = 0

    i = 0
    while i < n:
        sols = matchbox(sols, i)

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
        if clights == 0:
            # print("blackout night", seated_pairs)
            sols = [
                s for s in sols if not any(p in s for p in seated_pairs)
            ]  # filter out solutions with blackout pairs

        # filter solution by lights
        sols = [s for s in sols if len(seated_pairs & s) == clights]
        sols = [s for s in sols if set(known_pms).issubset(s)]

        # print(f"i = {i}, len(sols) = {len(sols)} after {clights} lights")
        i += 1

    # df = sols_as_df(sols)

    season = Season(
        "toy",
        lefts,
        rights,
        tuple(nights),
        matchboxes_from_list(mbs),
        mm_person=dm if with_dm else None,
        matchbox_reveals_dm=True,
        solution=solution,
        mm_known_from_week=0,
    )

    return season, sols


def check_generated_season(n: int = 5, with_dm: bool = True):
    season, sols = simulate_season(n, with_dm)
    solver = TupleSolver(season)

    solution = season.solution
    assert solution

    for l, r in solution:
        ## todo: check include_night
        if solver.no_match(l, r, n - 1, True):
            print(season)
            assert False, "no match in solution"

    for s in sols:
        if not solver.solution_possible(s, n - 1, True):
            print(season)
            assert False, (
                f"solution in sols not possible {solver.solution_possible(s, n - 1, True)}"
            )

    if len(sols) != len(solver.solve(n - 1)):
        print(season)
        assert False, (
            f"len(sols) = {len(sols)} != {len(solver.solve(n - 1))} = len(solver.solve(n - 1))"
        )

    if not solver.solution_possible(sols[0], n - 1, True):
        print(season)
        assert False, "solution not possible"

    fsols = {
        s
        for s in solver.generate_complete_solutions(frozenset(), n - 1, True)
        if solver.solution_possible(s, n - 1, True)
    }
    if len(sols) != len(fsols):
        print(season)
        assert False, "len(sols) != len(fsols)"

    for i in range(n - 1):
        fsols = {
            s
            for s in solver.generate_complete_solutions(frozenset(), i, True)
            if solver.solution_possible(s, i, True)
        }
        if len(fsols) != len(solver.solve(i)):
            print(season)
            assert False, "len(fsols) != len(solver.solve(i))"


if __name__ == "__main__":
    # uv run python -m scripts.artificial_seasons
    season, _ = simulate_season(5)
    # print(season.get_matchboxes(7, False))
    solver = TupleSolver(season)
    sols = solver.solve(0, False)
    print(len(sols))
    print(solver.season.get_matchboxes(0, False))
