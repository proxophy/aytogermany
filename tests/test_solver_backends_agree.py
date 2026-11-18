# By Claude

import random
import unittest
from itertools import islice

from ayto import Pair, Season, get_season, get_solver
from ayto.bitmask_solver import BitMaskSolver
from ayto.tuple_solver import TupleSolver
from tests.fixtures import ALL_TOY_SEASONS, BITMASK_SOLVERS, TUPLE_SOLVERS

type Pairs = frozenset[Pair]
type Mismatch = tuple[object, ...]

EPISODES = range(6)
INCLUDE_NIGHT = (False, True)
POOL_SIZE = 300
MAX_REPORTED = 5


def build_solvers(season: Season) -> tuple[TupleSolver, BitMaskSolver]:
    return (
        TUPLE_SOLVERS[season.match_model](season),
        BITMASK_SOLVERS[season.match_model](season),
    )


def candidate_solutions(
    season: Season, tuple_solver: TupleSolver, seed: int
) -> list[Pairs]:
    """Mix of complete solutions (mostly valid) and perturbed matchings
    (mostly invalid in interesting ways: extra/missing pairs, double matches)."""
    rng = random.Random(seed)
    n_lefts, n_rights = len(season.lefts_names), len(season.rights_names)
    all_pairs = [(l, r) for l in range(n_lefts) for r in range(n_rights)]

    empty_solver = type(tuple_solver)(season.as_empty())
    complete = [
        frozenset(s)
        for s in islice(
            empty_solver.generate_complete_solutions(
                empty_solver.from_pairs(()), 5, True
            ),
            2000,
        )
    ]
    pool = rng.sample(complete, min(POOL_SIZE // 2, len(complete)))

    for _ in range(POOL_SIZE // 2):
        rights = rng.sample(range(n_rights), min(n_lefts, n_rights))
        base = set(zip(range(n_lefts), rights))
        base |= set(rng.sample(all_pairs, rng.randint(0, 3)))
        for pair in rng.sample(sorted(base), rng.randint(0, 2)):
            base.discard(pair)
        pool.append(frozenset(base))
    return pool


class TestBackendsAgree(unittest.TestCase):
    def assertNoMismatches(self, mismatches: list[Mismatch]) -> None:
        self.assertFalse(
            mismatches,
            f"{len(mismatches)} disagreements (tuple, bitmask); first " +
            f"{MAX_REPORTED}: {mismatches[:MAX_REPORTED]}",
        )

    def test_violations_agree_on_toy_seasons(self):
        for seed, (name, season) in enumerate(ALL_TOY_SEASONS.items()):
            with self.subTest(season=name):
                tup, bit = build_solvers(season)
                mismatches: list[Mismatch] = []
                for sol in candidate_solutions(season, tup, seed):
                    tsol, bsol = tup.from_pairs(sol), bit.from_pairs(sol)
                    t, b = (
                        tup.find_format_violation(tsol),
                        bit.find_format_violation(bsol),
                    )
                    if t != b:
                        mismatches.append(("format", sorted(sol), t, b))
                    for episode in EPISODES:
                        t = tup.find_multiple_match_violation(tsol, episode)
                        b = bit.find_multiple_match_violation(bsol, episode)
                        if t != b:
                            mismatches.append(
                                ("multiple_match", episode, sorted(sol), t, b)
                            )
                        for icl in INCLUDE_NIGHT:
                            t = tup.find_violation(tsol, episode, icl)
                            b = bit.find_violation(bsol, episode, icl)
                            if t != b:
                                mismatches.append(
                                    ("violation", episode, icl, sorted(sol), t, b)
                                )
                self.assertNoMismatches(mismatches)

    def test_no_match_reason_agrees_on_toy_seasons(self):
        for name, season in ALL_TOY_SEASONS.items():
            with self.subTest(season=name):
                tup, bit = build_solvers(season)
                self._test_no_match_mismatches(tup, bit, season)

    def test_no_match_reason_agrees_on_real_seasons(self):
        for name in ("normalo2020", "normalo2024", "normalo2026", "vip2023"):
            with self.subTest(season=name):
                tup, bit = get_solver(name), get_solver(name, True)
                self._test_no_match_mismatches(tup, bit, tup.season)

    def test_normalo2020_nomatches(self):
        self._test_no_match_reason_agree_on_real_season("normalo2020")

    def test_normalo2021_nomatches(self):
        self._test_no_match_reason_agree_on_real_season("normalo2021")

    def test_normalo2022_nomatches(self):
        self._test_no_match_reason_agree_on_real_season("normalo2022")

    def test_normalo2023_nomatches(self):
        self._test_no_match_reason_agree_on_real_season("normalo2023")

    def test_normalo2024_nomatches(self):
        self._test_no_match_reason_agree_on_real_season("normalo2024")

    def test_normalo2025_nomatches(self):
        self._test_no_match_reason_agree_on_real_season("normalo2025")

    def test_normalo2026_nomatches(self):
        self._test_no_match_reason_agree_on_real_season("normalo2026")

    def test_vip2021_nomatches(self):
        self._test_no_match_reason_agree_on_real_season("vip2021")

    def test_vip2022_nomatches(self):
        self._test_no_match_reason_agree_on_real_season("vip2022")

    def test_vip2023_nomatches(self):
        self._test_no_match_reason_agree_on_real_season("vip2023")

    def test_vip2024_nomatches(self):
        self._test_no_match_reason_agree_on_real_season("vip2024")

    def test_vip2025_nomatches(self):
        self._test_no_match_reason_agree_on_real_season("vip2025")

    def test_vip2026_nomatches(self):
        self._test_no_match_reason_agree_on_real_season("vip2026")

    def _test_no_match_mismatches(
        self, tup: TupleSolver, bit: BitMaskSolver, season: Season
    ):
        mismatches: list[Mismatch] = []

        for episode in range(len(season.nights) + 1):
            for icl in INCLUDE_NIGHT:
                with self.subTest(episode=episode, include_night=icl):
                    for left in season.lefts_names:
                        for right in season.rights_names:
                            t = tup.str_no_match_reason(left, right, episode, icl)
                            b = bit.str_no_match_reason(left, right, episode, icl)
                            if t != b:
                                mismatches.append((left, right, episode, icl, t, b))
                    self.assertNoMismatches(mismatches)

    def _test_no_match_reason_agree_on_real_season(self, sn: str):
        season = get_season(sn)
        bit: BitMaskSolver = get_solver(sn, True)
        tup: TupleSolver = get_solver(sn, False)
        self._test_no_match_mismatches(tup, bit, season)

    def test_solve_agrees_on_toy_seasons(self):
        for name, season in ALL_TOY_SEASONS.items():
            with self.subTest(season=name):
                tup, bit = build_solvers(season)
                mismatches: list[Mismatch] = []
                for end in range(6):
                    for icl in INCLUDE_NIGHT:
                        t = set(tup.solve(end, icl))
                        b = {
                            frozenset(bit.codec.decode(s)) for s in bit.solve(end, icl)
                        }
                        if t != b:
                            mismatches.append((end, icl, len(t), len(b), len(t ^ b)))
                self.assertNoMismatches(mismatches)


if __name__ == "__main__":
    unittest.main()  # pyright: ignore[reportUnusedCallResult]
