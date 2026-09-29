# pyright: standard

import json
import random
import string
import sys
from collections import Counter
from typing import Any

from ayto import Solver
from ayto.analysis import *
from ayto.models import *
from ayto.utils import *


def comparetoao(solver: Solver, sols: list[Solution], end: int):
    with open("aodata.txt", "r") as f:
        lines = map(eval, f.readlines())

    def parse_sol(ld: dict[str, list[str]]):
        pairs: set[Pair] = set()
        for l, rs in ld.items():
            for r in rs:
                pairs.add((l, r))
        return Solution(set(pairs))

    solsao = list(map(parse_sol, lines))

    print("lengths", len(solsao), len(sols))
    count = count2 = 0
    reasons: list[str] = []
    for s in solsao:
        spd = solver.find_violation(s, end, True)
        if spd is not None:
            count += 1
            reasons.append(spd)
        elif s not in sols:
            count2 += 1
            print(s)
            # break

    print(
        f"solution in solsao not possible: {count}; solutions in solsao not in sols {count2}"
    )
    print(len(reasons), Counter(reasons))

    count = count2 = 0
    reasons = []
    for s in sols:
        if s not in solsao:
            spd = solver.find_violation(s, end, False)
            count += 1
            # print(s)
            if spd is not None:
                reasons.append(spd)
                # print(spd)
                count2 += 1
    print(Counter(reasons))
    print(f"sols not in solsao: {count}; {count2}")
    # print(Counter(nomatchp))


def check_ao_data(sols: list[Solution], end, sn):

    with open("ayto_data.json", encoding="utf-8") as f:
        all_data: dict[str, Any] = json.load(f)  # pyright: ignore[reportExplicitAny]

    season_data: dict[str, Any] = all_data[sn]  # pyright: ignore[reportExplicitAny, reportAny]
    weeks: dict[str, Any] = season_data["weeks"]  # pyright: ignore[reportExplicitAny, reportAny]
    # print("end", end)
    for week in weeks[: end + 1]:  # pyright: ignore[reportArgumentType]
        # print("week number", week["number"])
        for event in week["events"]:
            if event["type"] == "box":
                a, b = event["pair"]
                if event["result"] == "yes" and not (a, b) in sols:
                    return False, event
                if event["result"] == "no" and (a, b) in sols:
                    return False, event
            elif event["type"] == "matching_night":
                lights = sum(1 for a, b in event["pairs"] if (a, b) in sols)
                if lights != event["lights"]:
                    return False, event
            # else:
            #     print("other event", event)
    return True, None


def iterate_through(solver, start=3):
    amounts = []
    for end in range(start, 10):
        sols = solver.solve(end, True)
        amounts.append(len(sols))
    return amounts


def check_solution_possible(sols, solver: Solver, end):
    reasons = []
    for s in sols:
        p = solver.find_violation(s, end, True)
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


def toy_example():

    toy_nights = (
        Night(
            {
                ("A", "1"),
                ("B", "3"),
                ("C", "4"),
                ("D", "5"),
                ("E", "2"),
            },
            1,
        ),
        Night(
            {
                ("A", "5"),
                ("B", "2"),
                ("C", "3"),
                ("D", "4"),
                ("E", "1"),
            },
            3,
        ),
        Night(
            {
                ("A", "5"),
                ("B", "2"),
                ("C", "3"),
                ("D", "4"),
                ("E", "6"),
            },
            4,
        ),
    )
    toy_matchboxes = Matchboxes((0,), ((("B", "3")),), (False,))
    toy_lefts = ("A", "B", "C", "D", "E")
    toy_rights = ("1", "2", "3", "4", "5", "6")
    toy_season = Season(
        "toy", toy_lefts, toy_rights, toy_nights, toy_matchboxes, mm="6", num_matches=6
    )
    solver = Solver(toy_season)

    sols = solver.solve(5)
    print(len(sols))


def simulate_season(n: int = 5, with_dm: bool = True) -> tuple[Season, list[Solution]]:

    def matchbox(sols, i):
        mp = (random.choice(lefts), random.choice(rights))
        if mp in solution:
            sols = [s for s in sols if mp in s]
            known_pms.append(mp)
        else:
            sols = [s for s in sols if mp not in s]

        res = mp in solution
        mbs.append((i, mp, res))
        # reveal double match
        if res and mp[0] == "A":
            mr = next(iter({"1", dm} - {mp[1]}))
            np = ("A", mr)
            mbs.append((i, np, True))
            sols = [s for s in sols if np in s]
            known_pms.append(np)
            # print("double match", mp, mr)
        elif res:
            # double not possible with this l
            sols = [s for s in sols if (mp[0], dm) not in s]
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
    solution = frozenset({(l, r) for l, r in zip(lefts, rights[:-1])} | {("A", dm)})

    empty_season = Season(
        "toy",
        lefts,
        rights,
        (),
        Matchboxes(),
        mm=dm if with_dm else None,
        num_matches=n + 1,
        mm_known_after_week=0,
    )
    emptysolver = Solver(empty_season)

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
        mm=dm if with_dm else None,
        num_matches=n + 1,
        mb_reveals_dm=True,
        solution=solution,
        mm_known_after_week=0,
    )

    return season, sols


def check_generated_season(n: int = 5, with_dm: bool = True):
    season, sols = simulate_season(n, with_dm)
    solver = Solver(season)

    solution = season.solution
    assert solution

    for l, r in solution:
        if solver.no_match(l, r, n - 1):
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
    sn = "normalo2020"

    import cProfile

    profiler = cProfile.Profile()
    profiler.enable()

    season: Season = get_season(sn)
    # sol: Solution = season.solution 
    solver: Solver = get_solver(sn)  
    end = 6
    sols = solver.solve(end)
    print(len(sols))
    # plot_probs(sols)
    # print(solver.no_match("Julian M", "Linda", 4, True))

    sys.exit()

    import tqdm  # noqa: F401

    # for i in tqdm.trange(0, 100):
    #     check_generated_season(5, with_dm= True)

    # comparetoao(solver, sols, end)
    # analyze_solutions(sols)
    # print(check_solution_possible(sols, solver, end))
    # plot_probs(sols)

    # for psol in solver.generate_partial_solutions(end, True):
    #     if psol <= sol:
    #         print(psol)

    # sols = solver.generate_complete_solutions(psol, end, True)
    # print("sol ins sols", sol in psol)

    profiler.disable()
    import pstats

    p = pstats.Stats(profiler)
    p.sort_stats("cumulative").print_stats(5)
