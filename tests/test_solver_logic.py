import unittest

from ayto import (
    DoubleMatchForRightSolver,
    DoubleMatchPairSolver,
    TripleMatchSolver,
    TupleSolver,
    TwoDoubleMatchesSolver,
)
from ayto.base import NoMatchReason, Violation
from ayto.season import Matchboxes, Night, Season
from ayto.utils import get_solver

normalo2020 = get_solver("normalo2020")
normalo2024 = get_solver("normalo2024")
normalo2026 = get_solver("normalo2026")
vip2023 = get_solver("vip2023")
toy_nights = (
    Night({(0, 0), (1, 2), (2, 3), (3, 4), (4, 1)}, 1),
    Night({(0, 4), (1, 1), (2, 2), (3, 3), (4, 0)}, 3),
    Night({(0, 4), (1, 1), (2, 2), (3, 3), (4, 5)}, 4),
)
toy_matchboxes = Matchboxes((0,), ((1, 2),), (False,), (True,))
toy_lefts = ("A", "B", "C", "D", "E")
toy_rights = ("1", "2", "3", "4", "5", "6")
toy_season = Season(
    "toy", toy_lefts, toy_rights, toy_nights, toy_matchboxes, mm_person="6"
)


toy_season2 = Season(
    name="toy",
    lefts_names=("A", "B", "C", "D", "E"),
    rights_names=("1", "2", "3", "4", "5", "6"),
    nights=(
        Night(pairs={(4, 3), (0, 2), (1, 1), (3, 0), (2, 4)}, lights=1),
        Night(pairs={(0, 5), (1, 2), (4, 1), (3, 0), (2, 4)}, lights=1),
        Night(pairs={(2, 1), (1, 2), (4, 3), (3, 0), (0, 4)}, lights=0),
        Night(pairs={(3, 1), (1, 0), (4, 3), (2, 2), (0, 4)}, lights=1),
        Night(pairs={(4, 2), (2, 0), (1, 1), (3, 3), (0, 4)}, lights=2),
    ),
    matchboxes=Matchboxes(
        weeks=(0, 1, 3, 4),
        pairs=((0, 1), (3, 1), (3, 0), (2, 4)),
        results=(False, False, False, False),
        before_night=(
            True,
            True,
            True,
            True,
        ),
    ),
    solution=frozenset({(0, 0), (0, 5), (4, 4), (2, 2), (1, 1), (3, 3)}),
    mm_person="6",
    double_match_pair_known=7,
    mm_known_from_week=0,
    matchbox_reveals_dm=True,
)

toy_season_wo_dm = Season(
    "toy_wo_dm", toy_lefts, toy_rights, toy_nights, toy_matchboxes
)
toy_season_dm_later_known = Season(
    "toy_dm_later_known",
    toy_lefts,
    toy_rights,
    toy_nights,
    toy_matchboxes,
    mm_person="6",
    mm_known_from_week=2,
)

toy_season_with_double_match_pair = Season(
    "toy_with_dm_pair",
    toy_lefts,
    toy_rights,
    toy_nights,
    Matchboxes((0,), ((2, 0),), (False,), (True,)),
    match_model="double_match_pair",
    mm_person="6",
    double_match_pair_names=("1", "6"),
    double_match_pair_known=0,
)

toy_season_with_triple_match = Season(
    "toy",
    toy_lefts,
    ("1", "2", "3", "4", "5", "6", "7"),
    toy_nights,
    toy_matchboxes,
    match_model="triple_match",
    mm_person="6",
)

toy_season_with_two_double_matches = Season(
    "toy",
    toy_lefts,
    ("1", "2", "3", "4", "5", "6", "7"),
    toy_nights,
    toy_matchboxes,
    match_model="two_double_matches",
    mm_person="6",
)

toy_season_with_dm_for_right = Season(
    "toy",
    ("A", "B", "C", "D", "E", "F"),
    ("1", "2", "3", "4", "5", "6"),
    toy_nights,
    toy_matchboxes,
    match_model="double_match_for_right",
    extra_left_name="F",
)

model_to_toy = {
    "double_match": toy_season,
    "double_match_pair": toy_season_with_double_match_pair,
    "two_double_matches": toy_season_with_two_double_matches,
    "double_match_for_right": toy_season_with_dm_for_right,
    "triple_match": toy_season_with_triple_match,
}


class TestSolverLogic(unittest.TestCase):
    def test_no_match_matchbox(self):
        self.assertEqual(
            normalo2020.str_no_match_reason("Ivana", "Mo", 2, True),
            NoMatchReason.MATCHBOX_FALSE,
        )

    def test_no_match_matchbox_end(self):
        self.assertIsNone(normalo2020.str_no_match_reason("Luisa", "Ferhat", 2, True))
        self.assertEqual(
            normalo2020.str_no_match_reason("Luisa", "Ferhat", 10, True),
            NoMatchReason.MATCHBOX_FALSE,
        )

    def test_no_match_bo_night(self):
        self.assertEqual(
            normalo2020.str_no_match_reason("Nadine", "Dominic", 10, True),
            NoMatchReason.SITTING_IN_BLACKOUT_NIGHT,
        )

    def test_no_match_bo_night_end(self):
        self.assertIsNone(
            normalo2020.str_no_match_reason("Madleine", "Ferhat", 2, True)
        )
        self.assertEqual(
            normalo2020.str_no_match_reason("Madleine", "Ferhat", 3, True),
            NoMatchReason.SITTING_IN_BLACKOUT_NIGHT,
        )

    def test_no_match_double_match(self):
        self.assertEqual(
            normalo2020.str_no_match_reason("Sabrina", "Mo", 2, True),
            NoMatchReason.MEMBER_HAS_PARTNER_ALREADY,
        )
        self.assertEqual(
            normalo2020.str_no_match_reason("Aline", "Ferhat", 2, True),
            NoMatchReason.MEMBER_HAS_PARTNER_ALREADY,
        )
        self.assertIsNone(normalo2020.str_no_match_reason("Aline", "Mo", 2, True))

    def test_no_match_double_match_pair(self):
        self.assertEqual(
            vip2023.str_no_match_reason("Sabrina", "Max", 7, True),
            NoMatchReason.MATCHBOX_FALSE,
        )
        self.assertEqual(
            vip2023.str_no_match_reason("Sabrina", "Peter", 7, True),
            NoMatchReason.MATCHBOX_FALSE_FOR_DOUBLE_MATCH_PAIR,
        )

        self.assertEqual(
            vip2023.str_no_match_reason("Paulina", "Emanuell", 8, True),
            NoMatchReason.MEMBER_HAS_PARTNER_ALREADY,
        )

    def test_no_match_triple_match(self):
        self.assertIsNone(normalo2024.str_no_match_reason("Gerrit", "Edda", 3, True))
        self.assertIsNone(normalo2024.str_no_match_reason("Gerrit", "Pia", 3, True))
        self.assertEqual(
            normalo2024.str_no_match_reason("Gerrit", "Edda", 4, True),
            NoMatchReason.CANNOT_BE_IN_TRIPLE_MATCH,
        )

    def test_no_match_dm_later_known(self):
        self.assertIsNone(normalo2026.str_no_match_reason("Noel", "Alicia", 5, True))
        self.assertIsNone(normalo2026.str_no_match_reason("Noel", "Alicia", 6, True))
        self.assertEqual(
            normalo2026.str_no_match_reason("Julian M", "Alicia", 4, True),
            NoMatchReason.MATCHBOX_FALSE,
        )
        self.assertEqual(
            normalo2026.str_no_match_reason("Julian M", "Linda", 4, True),
            NoMatchReason.L_CANNOT_HAVE_SECOND_MATCH,
        )
        self.assertEqual(
            normalo2026.str_no_match_reason("Julian M", "Linda", 6, True),
            NoMatchReason.L_CANNOT_HAVE_SECOND_MATCH,
        )

    def test_solution_correct_format(self):
        solver = TupleSolver(toy_season)
        sol1 = {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 5)}

        self.assertIsNone(solver.find_format_violation(sol1))
        sol2 = sol1 | {(1, 2)}
        self.assertEqual(solver.find_format_violation(sol2), Violation.TOO_MANY_MATCHES)
        self.assertEqual(
            solver.find_format_violation({(0, 0), (1, 0)}),
            Violation.DOUBLE_MATCH_FOR_RIGHT,
        )

    def test_solution_possible_reason_no_match(self):
        solver = TupleSolver(toy_season)
        # (1,2) is false by matchbox
        sol1 = {(0, 0), (1, 2), (2, 1), (3, 3), (4, 4), (4, 5)}
        self.assertEqual(
            solver.find_violation(sol1, 1, True),
            Violation.CONTAINS_KNOWN_NO_MATCH,
        )

    def test_solution_possible_reason_lights(self):
        solver = TupleSolver(toy_season)
        sol1 = {(0, 0), (1, 1), (2, 3), (3, 4), (4, 2), (4, 5)}
        self.assertEqual(
            solver.find_violation(sol1, 1, True), Violation.TOO_MANY_LIGHTS_IN_NIGHT
        )
        sol2 = {(0, 3), (1, 1), (2, 4), (3, 5), (4, 2), (3, 0)}
        self.assertEqual(
            solver.find_violation(sol2, 1, True), Violation.NOT_ENOUGH_LIGHTS_IN_NIGHT
        )
        sol3 = {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 5)}
        self.assertIsNone(solver.find_violation(sol3, 2, True))

    def test_check_multiple_match_logic(self):
        solver = TupleSolver(toy_season)
        sol0 = {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 5)}
        self.assertIsNone(solver.find_multiple_match_violation(sol0, 2))
        sol1 = {(2, 2), (2, 3), (4, 4), (4, 5)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol1, 1),
            Violation.TOO_MANY_MULTIPLE_MATCHES,
        )
        # mm: index 5
        sol2 = {(3, 3), (3, 4)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 1),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )

        sol2 = {(3, 3), (3, 4), (1, 5)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 1),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )

        sol3 = {(0, 0), (0, 1), (0, 2)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol3, 1),
            Violation.MULTIPLE_MATCH_TOO_BIG,
        )
        sol4 = {(0, 0), (1, 0)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol4, 1),
            Violation.DOUBLE_MATCH_FOR_RIGHT,
        )

    def test_check_multiple_match_logic_wo_dm(self):
        # season without dm specified
        solver = TupleSolver(toy_season_wo_dm)
        sol0 = {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 5)}
        self.assertIsNone(solver.find_multiple_match_violation(sol0, 2))
        sol1 = {(2, 2), (2, 3), (4, 4), (4, 5)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol1, 1),
            Violation.TOO_MANY_MULTIPLE_MATCHES,
        )
        sol2 = {(0, 0), (0, 1), (0, 2)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 1),
            Violation.MULTIPLE_MATCH_TOO_BIG,
        )

        sol3 = {(3, 3), (3, 4)}
        self.assertIsNone(solver.find_multiple_match_violation(sol3, 1))

        sol4 = {(3, 3), (3, 4), (1, 5)}
        self.assertIsNone(solver.find_multiple_match_violation(sol4, 1))

    def test_check_multiple_match_logic_dm_later_known(self):
        solver = TupleSolver(toy_season_dm_later_known)
        # dm known after episode 2
        sol1 = {(3, 3), (3, 4)}
        self.assertIsNone(solver.find_multiple_match_violation(sol1, 1))
        self.assertEqual(
            solver.find_multiple_match_violation(sol1, 2),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )

        sol2 = {(3, 3), (3, 4), (1, 5)}
        self.assertIsNone(solver.find_multiple_match_violation(sol2, 1))
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 2),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )

    def test_check_multiple_match_logic_with_double_match_pair(self):
        solver = DoubleMatchPairSolver(toy_season_with_double_match_pair)
        sol0 = {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 5)}
        self.assertIsNone(solver.find_multiple_match_violation(sol0, 2))
        # double match pair indices: 0,5
        sol1 = {(3, 3), (3, 4)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol1, 3),
            Violation.DOUBLE_MATCH_LEFTS_NOT_DOUBLE_MATCH_PAIR,
        )
        sol2 = {(0, 0), (1, 5)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 3),
            Violation.DOUBLE_MATCH_PAIR_DO_NOT_HAVE_SAME_MATCH,
        )
        sol3 = {(0, 0), (0, 1), (1, 5)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol3, 3),
            Violation.DOUBLE_MATCH_PAIR_DO_NOT_HAVE_SAME_MATCH,
        )

    def test_check_multiple_match_logic_with_triple_match(self):
        solver = TripleMatchSolver(toy_season_with_triple_match)
        sol1 = {(0, 0), (0, 1), (0, 5)}
        self.assertIsNone(solver.find_multiple_match_violation(sol1, 1))
        sol2 = {(0, 0), (0, 1), (0, 2)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 1),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )
        sol3 = {(0, 0), (0, 1), (1, 5)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol3, 1),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )
        sol4 = {(0, 0), (0, 1), (0, 5), (0, 4)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol4, 1),
            Violation.MULTIPLE_MATCH_TOO_BIG,
        )
        sol5 = {(0, 0), (0, 1), (0, 5), (1, 2), (1, 3)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol5, 1),
            Violation.TOO_MANY_MULTIPLE_MATCHES,
        )
        sol6 = {(0, 0), (0, 1), (1, 2), (1, 5)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol6, 1),
            Violation.TOO_MANY_MULTIPLE_MATCHES,
        )

    def test_check_multiple_match_logic_with_two_double_matches(self):
        solver = TwoDoubleMatchesSolver(toy_season_with_two_double_matches)

        sol1 = {(0, 1), (0, 2), (1, 3), (1, 5)}
        self.assertIsNone(solver.find_multiple_match_violation(sol1, 1))
        sol2 = {(0, 1), (0, 2), (1, 3), (1, 5), (2, 0), (2, 4)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 1),
            Violation.TOO_MANY_MULTIPLE_MATCHES,
        )
        sol3 = {(0, 1), (0, 2), (0, 4), (1, 3), (1, 5)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol3, 1),
            Violation.MULTIPLE_MATCH_TOO_BIG,
        )
        sol4 = {(0, 1), (0, 2), (0, 4)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol4, 1),
            Violation.MULTIPLE_MATCH_TOO_BIG,
        )
        sol5 = {(0, 1), (0, 2), (1, 0), (1, 4)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol5, 1),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )
        sol6 = {(0, 1), (0, 2), (1, 0), (1, 4), (2, 5)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol6, 1),
            Violation.MM_NOT_IN_MULTIPLE_MATCH,
        )

    def test_check_multiple_match_logic_with_double_match_for_right(self):
        solver = DoubleMatchForRightSolver(toy_season_with_dm_for_right)
        sol1 = {(0, 1), (5, 1)}
        self.assertIsNone(solver.find_multiple_match_violation(sol1, 1))
        sol2 = {(0, 1), (4, 1)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 1),
            Violation.EXTRA_LEFT_NOT_IN_DOUBLE_RIGHT_MATCH,
        )
        sol3 = {(0, 1), (5, 1), (1, 2), (2, 2)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol3, 1),
            Violation.MORE_THAN_ONE_DOUBLE_RIGHT_MATCH,
        )
        sol4 = {(0, 1), (0, 2), (5, 2)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol4, 1),
            Violation.DOUBLE_MATCHES_CANNOT_CROSS,
        )

    def test_solve_with_dm(self):
        solver = TupleSolver(toy_season)
        self.compare_solve_approaches(solver)

    def test_solve_with_dm2(self):
        solver = TupleSolver(toy_season2)
        self.compare_solve_approaches(solver)

    def test_solve_with_dm_later_known(self):
        solver = TupleSolver(toy_season_dm_later_known)
        self.compare_solve_approaches(solver)

    def test_solve_without_dm(self):
        solver = TupleSolver(toy_season_wo_dm)
        self.compare_solve_approaches(solver)

    def test_solve_with_double_match_pair(self):
        solver = DoubleMatchPairSolver(toy_season_with_double_match_pair)
        self.compare_solve_approaches(solver)

    def test_solve_with_triple_match(self):
        solver = TripleMatchSolver(toy_season_with_triple_match)
        self.compare_solve_approaches(solver, solver_type=TripleMatchSolver)

    def test_solve_with_two_double_matches(self):
        solver = TwoDoubleMatchesSolver(toy_season_with_two_double_matches)
        self.compare_solve_approaches(solver, solver_type=TwoDoubleMatchesSolver)

    def test_solve_with_dm_for_right(self):
        solver = DoubleMatchForRightSolver(toy_season_with_dm_for_right)
        self.compare_solve_approaches(solver, solver_type=DoubleMatchForRightSolver)

    def compare_solve_approaches(
        self,
        solver: TupleSolver,
        solver_type: type[TupleSolver] = TupleSolver,
    ):
        season = solver.season
        empty = season.as_empty()
        solver_for_empty = solver_type(empty)
        for end in range(6):
            for icl in [False, True]:
                sols = set(
                    solver_for_empty.generate_complete_solutions(frozenset(), end, icl)
                )
                sols = [s for s in sols if solver.solution_possible(s, end, icl)]
                self.assertEqual(len(sols), len(solver.solve(end, icl)))
