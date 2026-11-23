import unittest
from typing import Any

import pandas as pd

from ayto import NoMatchReason, Season, Violation
from ayto.base import SolutionType, Solver
from tests.fixtures import (
    toy_season,
    toy_season2,
    toy_season_dm_later_known,
    toy_season_with_dm_for_right,
    toy_season_with_double_match_pair,
    toy_season_with_triple_match,
    toy_season_with_two_double_matches,
    toy_season_wo_dm,
)

df = pd.read_csv("analytics/num_solutions_extended.csv", index_col=0)
# df.columns = df.columns.astype(int)
df.index = df.index.str.strip()
df.columns = df.columns.astype(str).str.strip()


type AnySolver[S: SolutionType] = Solver[S, Any]  # pyright: ignore[reportExplicitAny]


class SolverLogicTestsBase[S: SolutionType](unittest.TestCase):
    def make_solver(self, season: Season) -> AnySolver[S]:
        raise NotImplementedError

    def real_solver(self, name: str) -> AnySolver[S]:
        raise NotImplementedError

    def test_black_out_nights(self):
        self.assertEqual(
            self.real_solver("normalo2020").season.get_black_out_nights(), [3]
        )
        self.assertEqual(
            self.real_solver("normalo2021").season.get_black_out_nights(), [7]
        )
        self.assertEqual(
            self.real_solver("normalo2022").season.get_black_out_nights(), []
        )
        self.assertEqual(
            self.real_solver("normalo2023").season.get_black_out_nights(), [6, 7]
        )
        self.assertEqual(
            self.real_solver("normalo2024").season.get_black_out_nights(), []
        )
        self.assertEqual(
            self.real_solver("normalo2025").season.get_black_out_nights(), []
        )
        self.assertEqual(
            self.real_solver("normalo2026").season.get_black_out_nights(), []
        )
        self.assertEqual(self.real_solver("vip2021").season.get_black_out_nights(), [7])
        self.assertEqual(self.real_solver("vip2022").season.get_black_out_nights(), [])
        self.assertEqual(self.real_solver("vip2023").season.get_black_out_nights(), [7])
        self.assertEqual(self.real_solver("vip2024").season.get_black_out_nights(), [3])
        self.assertEqual(self.real_solver("vip2025").season.get_black_out_nights(), [])
        self.assertEqual(self.real_solver("vip2026").season.get_black_out_nights(), [])

    def test_no_match_matchbox(self):
        self.assertEqual(
            self.real_solver("normalo2020").str_no_match_reason("Ivana", "Mo", 2, True),
            NoMatchReason.MATCHBOX_FALSE,
        )

    def test_no_match_matchbox_end(self):
        self.assertIsNone(
            self.real_solver("normalo2020").str_no_match_reason(
                "Luisa", "Ferhat", 2, True
            )
        )
        self.assertEqual(
            self.real_solver("normalo2020").str_no_match_reason(
                "Luisa", "Ferhat", 10, True
            ),
            NoMatchReason.MATCHBOX_FALSE,
        )

    def test_no_match_bo_night(self):
        self.assertEqual(
            self.real_solver("normalo2020").str_no_match_reason(
                "Nadine", "Dominic", 10, True
            ),
            NoMatchReason.SITTING_IN_BLACKOUT_NIGHT,
        )

    def test_no_match_bo_night_end(self):
        self.assertIsNone(
            self.real_solver("normalo2020").str_no_match_reason(
                "Madleine", "Ferhat", 2, True
            )
        )
        self.assertEqual(
            self.real_solver("normalo2020").str_no_match_reason(
                "Madleine", "Ferhat", 3, True
            ),
            NoMatchReason.SITTING_IN_BLACKOUT_NIGHT,
        )

    def test_no_match_double_match(self):
        self.assertEqual(
            self.real_solver("normalo2020").str_no_match_reason(
                "Sabrina", "Mo", 2, True
            ),
            NoMatchReason.MEMBER_HAS_PARTNER_ALREADY,
        )
        self.assertEqual(
            self.real_solver("normalo2020").str_no_match_reason(
                "Aline", "Ferhat", 2, True
            ),
            NoMatchReason.MEMBER_HAS_PARTNER_ALREADY,
        )
        self.assertIsNone(
            self.real_solver("normalo2020").str_no_match_reason("Aline", "Mo", 2, True)
        )

    def test_no_match_double_match_pair(self):
        self.assertEqual(
            self.real_solver("vip2023").str_no_match_reason("Sabrina", "Max", 7, True),
            NoMatchReason.MATCHBOX_FALSE,
        )
        self.assertEqual(
            self.real_solver("vip2023").str_no_match_reason(
                "Sabrina", "Peter", 7, True
            ),
            NoMatchReason.MATCHBOX_FALSE_FOR_DOUBLE_MATCH_PAIR,
        )

        self.assertEqual(
            self.real_solver("vip2023").str_no_match_reason(
                "Paulina", "Emanuell", 8, True
            ),
            NoMatchReason.MEMBER_HAS_PARTNER_ALREADY,
        )

    def test_no_match_triple_match(self):
        self.assertIsNone(
            self.real_solver("normalo2024").str_no_match_reason(
                "Gerrit", "Edda", 3, True
            )
        )
        self.assertIsNone(
            self.real_solver("normalo2024").str_no_match_reason(
                "Gerrit", "Pia", 3, True
            )
        )
        self.assertEqual(
            self.real_solver("normalo2024").str_no_match_reason(
                "Gerrit", "Edda", 4, True
            ),
            NoMatchReason.CANNOT_BE_IN_TRIPLE_MATCH,
        )

    def test_no_match_dm_later_known(self):
        self.assertIsNone(
            self.real_solver("normalo2026").str_no_match_reason(
                "Noel", "Alicia", 5, True
            )
        )
        self.assertIsNone(
            self.real_solver("normalo2026").str_no_match_reason(
                "Noel", "Alicia", 6, True
            )
        )
        self.assertEqual(
            self.real_solver("normalo2026").str_no_match_reason(
                "Julian M", "Alicia", 4, True
            ),
            NoMatchReason.MATCHBOX_FALSE,
        )
        self.assertEqual(
            self.real_solver("normalo2026").str_no_match_reason(
                "Julian M", "Linda", 4, True
            ),
            NoMatchReason.L_CANNOT_HAVE_SECOND_MATCH,
        )
        self.assertEqual(
            self.real_solver("normalo2026").str_no_match_reason(
                "Julian M", "Linda", 6, True
            ),
            NoMatchReason.L_CANNOT_HAVE_SECOND_MATCH,
        )

    def test_solution_correct_format(self):
        solver = self.make_solver(toy_season)
        sol1 = solver.from_pairs({(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 5)})

        self.assertIsNone(solver.find_format_violation(sol1))
        sol2 = sol1 | solver.from_pairs({(1, 2)})
        self.assertEqual(solver.find_format_violation(sol2), Violation.TOO_MANY_MATCHES)
        self.assertEqual(
            solver.find_format_violation(solver.from_pairs({(0, 0), (1, 0)})),
            Violation.DOUBLE_MATCH_FOR_RIGHT,
        )

    def test_solution_possible_reason_no_match(self):
        solver = self.make_solver(toy_season)
        # (1,2) is false by matchbox
        sol1 = solver.from_pairs({(0, 0), (1, 2), (2, 1), (3, 3), (4, 4), (4, 5)})
        self.assertEqual(
            solver.find_violation(sol1, 1, True),
            Violation.CONTAINS_KNOWN_NO_MATCH,
        )

    def test_solution_possible_reason_lights(self):
        solver = self.make_solver(toy_season)
        sol1 = solver.from_pairs({(0, 0), (1, 1), (2, 3), (3, 4), (4, 2), (4, 5)})
        self.assertEqual(
            solver.find_violation(sol1, 1, True), Violation.TOO_MANY_LIGHTS_IN_NIGHT
        )
        sol2 = solver.from_pairs({(0, 3), (1, 1), (2, 4), (3, 5), (4, 2), (3, 0)})
        self.assertEqual(
            solver.find_violation(sol2, 1, True), Violation.NOT_ENOUGH_LIGHTS_IN_NIGHT
        )
        sol3 = solver.from_pairs({(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 5)})
        self.assertIsNone(solver.find_violation(sol3, 2, True))

    def test_check_multiple_match_logic(self):
        solver = self.make_solver(toy_season)
        sol0 = solver.from_pairs({(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 5)})
        self.assertIsNone(solver.find_multiple_match_violation(sol0, 2))
        sol1 = solver.from_pairs({(2, 2), (2, 3), (4, 4), (4, 5)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol1, 1),
            Violation.TOO_MANY_MULTIPLE_MATCHES,
        )
        # mm: index 5
        sol2 = solver.from_pairs({(3, 3), (3, 4)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 1),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )

        sol2 = solver.from_pairs({(3, 3), (3, 4), (1, 5)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 1),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )

        sol3 = solver.from_pairs({(0, 0), (0, 1), (0, 2)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol3, 1),
            Violation.MULTIPLE_MATCH_TOO_BIG,
        )
        sol4 = solver.from_pairs({(0, 0), (1, 0)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol4, 1),
            Violation.DOUBLE_MATCH_FOR_RIGHT,
        )

    def test_check_multiple_match_logic_wo_dm(self):
        # season without dm specified
        solver = self.make_solver(toy_season_wo_dm)
        sol0 = solver.from_pairs({(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 5)})
        self.assertIsNone(solver.find_multiple_match_violation(sol0, 2))
        sol1 = solver.from_pairs({(2, 2), (2, 3), (4, 4), (4, 5)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol1, 1),
            Violation.TOO_MANY_MULTIPLE_MATCHES,
        )
        sol2 = solver.from_pairs({(0, 0), (0, 1), (0, 2)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 1),
            Violation.MULTIPLE_MATCH_TOO_BIG,
        )

        sol3 = solver.from_pairs({(3, 3), (3, 4)})
        self.assertIsNone(solver.find_multiple_match_violation(sol3, 1))

        sol4 = solver.from_pairs({(3, 3), (3, 4), (1, 5)})
        self.assertIsNone(solver.find_multiple_match_violation(sol4, 1))

    def test_check_multiple_match_logic_dm_later_known(self):
        solver = self.make_solver(toy_season_dm_later_known)
        # dm known after episode 2
        sol1 = solver.from_pairs({(3, 3), (3, 4)})
        self.assertIsNone(solver.find_multiple_match_violation(sol1, 1))
        self.assertEqual(
            solver.find_multiple_match_violation(sol1, 2),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )

        sol2 = solver.from_pairs({(3, 3), (3, 4), (1, 5)})
        self.assertIsNone(solver.find_multiple_match_violation(sol2, 1))
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 2),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )

    def test_check_multiple_match_logic_with_double_match_pair(self):
        solver = self.make_solver(toy_season_with_double_match_pair)
        sol0 = solver.from_pairs({(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 5)})
        self.assertIsNone(solver.find_multiple_match_violation(sol0, 2))
        # double match pair indices: (0,5), dm: 5
        sol1 = solver.from_pairs({(3, 3), (3, 5)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol1, 3),
            Violation.DOUBLE_MATCH_LEFTS_NOT_DOUBLE_MATCH_PAIR,
        )
        sol2 = solver.from_pairs({(0, 0), (1, 5)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 3),
            Violation.DOUBLE_MATCH_PAIR_DO_NOT_HAVE_SAME_MATCH,
        )
        sol3 = solver.from_pairs({(0, 0), (0, 1), (1, 5)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol3, 3),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )

    def test_check_multiple_match_logic_with_triple_match(self):
        solver = self.make_solver(toy_season_with_triple_match)
        sol1 = solver.from_pairs({(0, 0), (0, 1), (0, 5)})
        self.assertIsNone(solver.find_multiple_match_violation(sol1, 1))
        sol2 = solver.from_pairs({(0, 0), (0, 1), (0, 2)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 1),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )
        sol3 = solver.from_pairs({(0, 0), (0, 1), (1, 5)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol3, 1),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )
        sol4 = solver.from_pairs({(0, 0), (0, 1), (0, 5), (0, 4)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol4, 1),
            Violation.MULTIPLE_MATCH_TOO_BIG,
        )
        sol5 = solver.from_pairs({(0, 0), (0, 1), (0, 5), (1, 2), (1, 3)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol5, 1),
            Violation.TOO_MANY_MULTIPLE_MATCHES,
        )
        sol6 = solver.from_pairs({(0, 0), (0, 1), (1, 2), (1, 5)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol6, 1),
            Violation.TOO_MANY_MULTIPLE_MATCHES,
        )

    def test_check_multiple_match_logic_with_two_double_matches(self):
        solver = self.make_solver(toy_season_with_two_double_matches)

        sol1 = solver.from_pairs({(0, 1), (0, 2), (1, 3), (1, 5)})
        self.assertIsNone(solver.find_multiple_match_violation(sol1, 1))
        sol2 = solver.from_pairs({(0, 1), (0, 2), (1, 3), (1, 5), (2, 0), (2, 4)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 1),
            Violation.TOO_MANY_MULTIPLE_MATCHES,
        )
        sol3 = solver.from_pairs({(0, 1), (0, 2), (0, 4), (1, 3), (1, 5)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol3, 1),
            Violation.MULTIPLE_MATCH_TOO_BIG,
        )
        sol4 = solver.from_pairs({(0, 1), (0, 2), (0, 4)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol4, 1),
            Violation.MULTIPLE_MATCH_TOO_BIG,
        )
        sol5 = solver.from_pairs({(0, 1), (0, 2), (1, 0), (1, 4)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol5, 1),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )
        sol6 = solver.from_pairs({(0, 1), (0, 2), (1, 0), (1, 4), (2, 5)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol6, 1),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )

    def test_check_multiple_match_logic_with_double_match_for_right(self):
        solver = self.make_solver(toy_season_with_dm_for_right)
        sol1 = solver.from_pairs({(0, 1), (5, 1)})
        self.assertIsNone(solver.find_multiple_match_violation(sol1, 1))
        sol2 = solver.from_pairs({(0, 1), (4, 1)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 1),
            Violation.EXTRA_LEFT_NOT_IN_DOUBLE_RIGHT_MATCH,
        )
        sol3 = solver.from_pairs({(0, 1), (5, 1), (1, 2), (2, 2)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol3, 1),
            Violation.MORE_THAN_ONE_DOUBLE_RIGHT_MATCH,
        )
        sol4 = solver.from_pairs({(0, 1), (0, 2), (5, 2)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol4, 1),
            Violation.DOUBLE_MATCHES_CANNOT_CROSS,
        )
        sol5 = solver.from_pairs({(0, 0), (1, 0), (5, 0)})
        self.assertEqual(
            solver.find_multiple_match_violation(sol5, 1),
            Violation.RIGHT_MULTIPLE_MATCH_TOO_BIG,
        )

    def test_solve_with_dm(self):
        solver = self.make_solver(toy_season)
        self.compare_solve_approaches(solver)

    def test_solve_with_dm2(self):
        solver = self.make_solver(toy_season2)
        self.compare_solve_approaches(solver)

    def test_solve_with_dm_later_known(self):
        solver = self.make_solver(toy_season_dm_later_known)
        self.compare_solve_approaches(solver)

    def test_solve_without_dm(self):
        solver = self.make_solver(toy_season_wo_dm)
        self.compare_solve_approaches(solver)

    def test_solve_with_double_match_pair(self):
        solver = self.make_solver(toy_season_with_double_match_pair)
        self.compare_solve_approaches(solver)

    def test_solve_with_triple_match(self):
        solver = self.make_solver(toy_season_with_triple_match)
        self.compare_solve_approaches(solver)

    def test_solve_with_two_double_matches(self):
        solver = self.make_solver(toy_season_with_two_double_matches)
        self.compare_solve_approaches(solver)

    def test_solve_with_dm_for_right(self):
        solver = self.make_solver(toy_season_with_dm_for_right)
        self.compare_solve_approaches(solver)

    def compare_solve_approaches(self, solver: AnySolver[S]):
        empty_solver = type(solver)(solver.season.as_empty())
        for end in range(6):
            for icl in (False, True):
                with self.subTest(end=end, icl=icl):
                    generated = set(
                        empty_solver.generate_complete_solutions(
                            empty_solver.from_pairs(()), end, icl
                        )
                    )
                    expected = {
                        solver.to_pairs(s)
                        for s in generated
                        if solver.solution_possible(s, end, icl)
                    }
                    solved = solver.solve(end, icl)
                    actual = {solver.to_pairs(s) for s in solved}
                    # also catches duplicates: solve() returns a list
                    self.assertEqual(len(expected), len(solved))
                    self.assertEqual(expected, actual)
                    # solve() skips validation by default; it must not matter
                    validated = {
                        solver.to_pairs(s)
                        for s in solver.solve(end, icl, validate=True)
                    }
                    self.assertEqual(validated, actual)


class NumSolTestBase[S: SolutionType](unittest.TestCase):
    def make_solver(self, season: Season) -> AnySolver[S]:
        raise NotImplementedError

    def real_solver(self, name: str) -> AnySolver[S]:
        raise NotImplementedError

    def test_normalo2020_extended(self):
        self.check_season_extended("normalo2020")

    def test_normalo2021_extended(self):
        self.check_season_extended("normalo2021")

    def test_normalo2022_extended(self):
        self.check_season_extended("normalo2022")

    def test_normalo2023_extended(self):
        self.check_season_extended("normalo2023")

    def test_normalo2024_extended(self):
        self.check_season_extended("normalo2024")

    def test_normalo2025_extended(self):
        self.check_season_extended("normalo2025")

    def test_normalo2026_extended(self):
        self.check_season_extended("normalo2026")

    def test_vip2021_extended(self):
        self.check_season_extended("vip2021")

    def test_vip2022_extended(self):
        self.check_season_extended("vip2022")

    def test_vip2023_extended(self):
        self.check_season_extended("vip2023")

    def test_vip2024_extended(self):
        self.check_season_extended("vip2024")

    def test_vip2025_extended(self):
        self.check_season_extended("vip2025", start=3)

    def test_vip2026_extended(self):
        self.check_season_extended("vip2026")

    def check_season_extended(self, sn: str, start: int = 2):
        solver = self.real_solver(sn)
        for i in range(start, 10):
            with self.subTest(episode=i, include_night=False):
                sols = solver.solve(i, include_night=False)

                self.assertEqual(
                    len(sols),
                    df.loc[sn, f"b{i}"],
                    f"{sn} i={i}: got {len(sols)}, expected {df.loc[sn, f'b{i}']}",
                )

                solsf = [s for s in sols if solver.solution_possible(s, i, False)]
                self.assertEqual(
                    len(sols),
                    len(solsf),
                    f"{sn} i={i}: number of solutions != number of filtered solutions",
                )
            with self.subTest(episode=i, include_night=True):
                sols = solver.solve(i, True)
                self.assertEqual(
                    len(sols),
                    df.loc[sn, str(i)],
                    f"{sn} i={i}: got {len(sols)}, expected {df.loc[sn, str(i)]}",
                )

                solsf = [s for s in sols if solver.solution_possible(s, i, True)]
                self.assertEqual(
                    len(sols),
                    len(solsf),
                    f"{sn} i={i}: number of solutions != number of filtered solutions",
                )
