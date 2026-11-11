import unittest

from ayto import DoubleMatchPairSolver, TupleSolver
from ayto.season import Matchboxes, Night, Season
from ayto.utils import get_solver

normalo2020 = get_solver("normalo2020")
normalo2024 = get_solver("normalo2024")
normalo2026 = get_solver("normalo2026")
vip2023 = get_solver("vip2023")
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
toy_matchboxes = Matchboxes((0,), ((1, 2),), (False,), (True,))
toy_lefts = ("A", "B", "C", "D", "E")
toy_rights = ("1", "2", "3", "4", "5", "6")
toy_season = Season(
    "toy", toy_lefts, toy_rights, toy_nights, toy_matchboxes, mm_person="6"
)
toy_season_wo_dm = Season(
    "toy_wo_dm", toy_lefts, toy_rights, toy_nights, toy_matchboxes
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
        weeks=(0, 1, 2, 3, 4),
        pairs=((0, 1), (3, 1), (3, 1), (3, 0), (2, 4)),
        results=(False, False, False, False, False),
        before_night=(
            True,
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

# todo: type this


class TestSeasonSolution(unittest.TestCase):
    def test_no_match_matchbox(self):
        self.assertEqual(
            normalo2020.str_no_match_reason("Ivana", "Mo", 2, True), "matchbox_false"
        )

    def test_no_match_matchbox_end(self):
        self.assertIsNone(normalo2020.str_no_match_reason("Luisa", "Ferhat", 2, True))
        self.assertEqual(
            normalo2020.str_no_match_reason("Luisa", "Ferhat", 10, True),
            "matchbox_false",
        )

    def test_no_match_bo_night(self):
        self.assertEqual(
            normalo2020.str_no_match_reason("Nadine", "Dominic", 10, True),
            "sitting_in_blackout_night",
        )

    def test_no_match_bo_night_end(self):
        self.assertIsNone(
            normalo2020.str_no_match_reason("Madleine", "Ferhat", 2, True)
        )
        self.assertEqual(
            normalo2020.str_no_match_reason("Madleine", "Ferhat", 3, True),
            "sitting_in_blackout_night",
        )

    def test_no_match_double_match(self):
        self.assertEqual(
            normalo2020.str_no_match_reason("Sabrina", "Mo", 2, True),
            "member_has_partner_already",
        )
        self.assertEqual(
            normalo2020.str_no_match_reason("Aline", "Ferhat", 2, True),
            "member_has_partner_already",
        )
        self.assertIsNone(normalo2020.str_no_match_reason("Aline", "Mo", 2, True))

    def test_no_match_double_match_pair(self):
        self.assertEqual(
            vip2023.str_no_match_reason("Sabrina", "Max", 7, True),
            "matchbox_false_for_double_match_pair",
        )
        self.assertEqual(
            vip2023.str_no_match_reason("Sabrina", "Peter", 7, True),
            "matchbox_false_for_double_match_pair",
        )

    def test_no_match_triple_match(self):
        self.assertIsNone(normalo2024.str_no_match_reason("Gerrit", "Edda", 3, True))
        self.assertIsNone(normalo2024.str_no_match_reason("Gerrit", "Pia", 3, True))
        self.assertEqual(
            normalo2024.str_no_match_reason("Gerrit", "Edda", 4, True),
            "triple_match_already_exists",
        )

    def test_no_match_dm_later_known(self):
        self.assertIsNone(normalo2026.str_no_match_reason("Noel", "Alicia", 5, True))
        self.assertIsNone(normalo2026.str_no_match_reason("Noel", "Alicia", 6, True))
        self.assertEqual(
            normalo2026.str_no_match_reason("Julian M", "Alicia", 4, True),
            "matchbox_false",
        )
        self.assertEqual(
            normalo2026.str_no_match_reason("Julian M", "Linda", 4, True),
            "cannot_be_in_double_match",
        )
        self.assertEqual(
            normalo2026.str_no_match_reason("Julian M", "Linda", 6, True),
            "l_cannot_have_double_match",
        )

    def test_solution_correct_format(self):
        solver = TupleSolver(toy_season)
        sol1 = {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 5)}

        # todo: specify return str
        self.assertIsNone(solver.find_format_violation(sol1))
        sol1.add((1, 2))
        self.assertTrue(solver.find_format_violation(sol1))
        self.assertTrue(solver.find_format_violation({(0, 0), (1, 0)}))
        # self.assertTrue(solver.find_format_violation({(0, 0), (0,0)}))

    def test_solution_possible_reason_no_match(self):
        solver = TupleSolver(toy_season)
        sol1 = {(0, 0), (1, 2), (2, 1), (3, 3), (4, 4), (4, 5)}
        self.assertEqual(
            solver.find_violation(sol1, 1, True),
            "known_no_match",
        )

    def test_solution_possible_reason_lights(self):
        solver = TupleSolver(toy_season)
        sol1 = {(0, 0), (1, 1), (2, 3), (3, 4), (4, 2), (4, 5)}
        self.assertEqual(
            solver.find_violation(sol1, 1, True), "too_many_lights_in_night"
        )
        sol2 = {(0, 3), (1, 1), (2, 4), (3, 5), (4, 2), (3, 0)}
        self.assertEqual(
            solver.find_violation(sol2, 1, True), "not_enough_lights_in_night"
        )
        sol3 = {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 5)}
        self.assertIsNone(solver.find_violation(sol3, 2, True))

    def test_check_multiple_match_logic(self):
        solver = TupleSolver(toy_season)
        sol1 = {
            (0, 0),
            (1, 1),
            (2, 2),
            (2, 3),
            (4, 4),
            (4, 5),
        }
        self.assertEqual(
            solver.find_multiple_match_violation(sol1, 1), "too_many_double_matches"
        )
        sol2 = {(0, 0), (1, 1), (2, 2), (3, 3), (3, 4), (4, 5)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 1),
            "mm_not_in_complete_multiple_match",
        )
        sol3 = {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }
        self.assertEqual(
            solver.find_multiple_match_violation(sol3, 1),
            "too_many_double_matches",
        )
        sol4 = {(0, 0), (0, 1), (0, 2)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol4, 1),
            "triple_match_not_allowed",
        )

    def test_solve_with_dm(self):
        solver = TupleSolver(toy_season)
        self.compare_solve_approaches(solver, end=2)

    def test_solve_with_dm2(self):
        solver = TupleSolver(toy_season2)
        self.compare_solve_approaches(solver, end=4)

    def test_check_multiple_match_logic_wo_dm(self):
        # season without dm specified
        solver = TupleSolver(toy_season_wo_dm)
        sol1 = {(0, 0), (1, 1), (2, 2), (3, 3), (3, 4), (4, 5)}
        self.assertFalse(solver.find_multiple_match_violation(sol1, 2))
        sol2 = {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 1),
            "too_many_double_matches",
        )
        sol3 = {(0, 0), (0, 1), (0, 2)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol3, 1),
            "triple_match_not_allowed",
        )

    def test_solve_without_dm(self):
        solver = TupleSolver(toy_season_wo_dm)
        self.compare_solve_approaches(solver)

    def test_check_multiple_match_logic_with_double_match_pair(self):
        solver = DoubleMatchPairSolver(toy_season_with_double_match_pair)
        sol1 = {(0, 0), (1, 1), (2, 2), (3, 3), (3, 4), (0, 5)}
        self.assertTrue(solver.find_multiple_match_violation(sol1, 3))
        sol2 = {
            (0, 0),
            (1, 5),
        }
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 3),
            "double_match_pair_not_same_match",
        )
        sol3 = {(0, 0), (0, 1)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol3, 3),
            "double_match_lefts_not_double_match_pair",
        )
        sol4 = {(0, 0), (0, 1), (0, 2)}
        self.assertEqual(
            solver.find_multiple_match_violation(sol4, 1),
            "triple_match_not_allowed",
        )

    def compare_solve_approaches(self, solver: TupleSolver, end: int = 5):
        season = solver.season
        empty_season = Season(
            "empty",
            season.lefts_names,
            season.rights_names,
            (),
            Matchboxes(),
            match_model=season.match_model,
            mm_person=season.mm_person,
            double_match_pair_names=season.double_match_pair_names,
            double_match_pair_known=season.double_match_pair_known,
            mm_known_from_week=season.mm_known_from_week,
            matchbox_reveals_dm=season.matchbox_reveals_dm,
        )
        empty_season = TupleSolver(empty_season)
        sols = set(empty_season.generate_complete_solutions(frozenset(), end, True))
        sols = [s for s in sols if solver.solution_possible(s, end, True)]
        self.assertEqual(len(sols), len(solver.solve(end)))
