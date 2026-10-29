import unittest

from ayto import Solver, VIP2023Solver
from ayto.models import Matchboxes, Night, Season
from ayto.utils import get_solver

normalo2020 = get_solver("normalo2020")
normalo2024 = get_solver("normalo2024")
normalo2026 = get_solver("normalo2026")
vip2023 = get_solver("vip2023")

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
toy_matchboxes = Matchboxes((0,), (("B", "3"),), (False,))
toy_lefts = ("A", "B", "C", "D", "E")
toy_rights = ("1", "2", "3", "4", "5", "6")
toy_season = Season(
    "toy", toy_lefts, toy_rights, toy_nights, toy_matchboxes, mm="6", num_matches=6
)
toy_season_wo_dm = Season(
    "toy_wo_dm", toy_lefts, toy_rights, toy_nights, toy_matchboxes, num_matches=6
)

toy_season_with_double_match_pair = Season(
    "toy_wo_dm",
    toy_lefts,
    toy_rights,
    toy_nights,
    Matchboxes((0,), (("B", "1"),), (False,)),
    mm="6",
    double_match_pair=("1", "6"),
    double_match_pair_known=0,
    num_matches=6,
)


toy_season2 = Season(
    name="toy",
    lefts=("A", "B", "C", "D", "E"),
    rights=("1", "2", "3", "4", "5", "6"),
    nights=(
        Night(
            pairs={("E", "4"), ("A", "3"), ("B", "2"), ("D", "1"), ("C", "5")}, lights=1
        ),
        Night(
            pairs={("A", "6"), ("B", "3"), ("E", "2"), ("D", "1"), ("C", "5")}, lights=1
        ),
        Night(
            pairs={("C", "2"), ("B", "3"), ("E", "4"), ("D", "1"), ("A", "5")}, lights=0
        ),
        Night(
            pairs={("D", "2"), ("B", "1"), ("E", "4"), ("C", "3"), ("A", "5")}, lights=1
        ),
        Night(
            pairs={("E", "3"), ("C", "1"), ("B", "2"), ("D", "4"), ("A", "5")}, lights=2
        ),
    ),
    matchboxes=Matchboxes(
        weeks=(0, 1, 2, 3, 4),
        pairs=(("A", "2"), ("D", "2"), ("D", "2"), ("D", "1"), ("C", "5")),
        results=(False, False, False, False, False),
    ),
    solution=frozenset(
        {("A", "1"), ("A", "6"), ("E", "5"), ("C", "3"), ("B", "2"), ("D", "4")}
    ),
    mm="6",
    num_matches=6,
    double_match_pair=None,
    double_match_pair_known=7,
    mm_known_after_week=0,
    max_mm_size=2,
    mb_reveals_dm=True,
)

# todo: type this

class TestSeasonSolution(unittest.TestCase):
    def test_no_match_matchbox(self):
        self.assertEqual(
            normalo2020.no_match_reason("Ivana", "Mo", 2), "matchbox_false"
        )

    def test_no_match_matchbox_end(self):
        self.assertIsNone(normalo2020.no_match_reason("Luisa", "Ferhat", 2))
        self.assertEqual(
            normalo2020.no_match_reason("Luisa", "Ferhat", 10), "matchbox_false"
        )

    def test_no_match_bo_night(self):
        self.assertEqual(
            normalo2020.no_match_reason("Nadine", "Dominic", 10),
            "sitting_in_blackout_night",
        )

    def test_no_match_bo_night_end(self):
        self.assertIsNone(normalo2020.no_match_reason("Madleine", "Ferhat", 2))
        self.assertEqual(
            normalo2020.no_match_reason("Madleine", "Ferhat", 3),
            "sitting_in_blackout_night",
        )

    def test_no_match_double_match(self):
        self.assertEqual(
            normalo2020.no_match_reason("Sabrina", "Mo", 2),
            "member_has_partner_already",
        )
        self.assertEqual(
            normalo2020.no_match_reason("Aline", "Ferhat", 2),
            "member_has_partner_already",
        )
        self.assertIsNone(normalo2020.no_match_reason("Aline", "Mo", 2))

    def test_no_match_double_match_pair(self):
        self.assertEqual(
            vip2023.no_match_reason("Sabrina", "Max", 7),
            "matchbox_false_for_double_match_pair",
        )
        self.assertEqual(
            vip2023.no_match_reason("Sabrina", "Peter", 7),
            "matchbox_false_for_double_match_pair",
        )

    def test_no_match_tripple_match(self):
        self.assertIsNone(normalo2024.no_match_reason("Gerrit", "Edda", 3))
        self.assertIsNone(normalo2024.no_match_reason("Gerrit", "Pia", 3))
        self.assertEqual(
            normalo2024.no_match_reason("Gerrit", "Edda", 4),
            "tripple_match_already_exists",
        )

    def test_no_match_dm_later_known(self):
        self.assertIsNone(normalo2026.no_match_reason("Noel", "Alicia", 5))
        self.assertIsNone(normalo2026.no_match_reason("Noel", "Alicia", 6))
        self.assertEqual(
            normalo2026.no_match_reason("Julian M", "Alicia", 4), "matchbox_false"
        )
        self.assertEqual(
            normalo2026.no_match_reason("Julian M", "Linda", 4),
            "cannot_be_in_double_match",
        )
        self.assertEqual(
            normalo2026.no_match_reason("Julian M", "Linda", 6),
            "l_cannot_have_double_match",
        )

    def test_solution_correct_format(self):
        solver = Solver(toy_season)
        sol1 = {("A", "1"), ("B", "2"), ("C", "3"), ("D", "4"), ("E", "5"), ("E", "6")}

        # todo: specify return str
        self.assertIsNone(solver.find_format_violation(sol1))
        sol1.add(("B", "3"))
        self.assertTrue(solver.find_format_violation(sol1))
        self.assertTrue(solver.find_format_violation({("A", "1"), ("B", "1")}))
        self.assertTrue(solver.find_format_violation({("A", "1"), ("B", "2s")}))
        self.assertTrue(solver.find_format_violation({("A", "1"), ("Bs", "2")}))

    def test_solution_possible_reason_no_match(self):
        solver = Solver(toy_season)
        sol1 = {("A", "1"), ("B", "3"), ("C", "2"), ("D", "4"), ("E", "5"), ("E", "6")}
        self.assertEqual(
            solver.find_violation(sol1, 1, True),
            "known_no_match",
        )

    def test_solution_possible_reason_lights(self):
        solver = Solver(toy_season)
        sol1 = {("A", "1"), ("B", "2"), ("C", "4"), ("D", "5"), ("E", "3"), ("E", "6")}
        self.assertEqual(
            solver.find_violation(sol1, 1, True), "too_many_lights_in_night"
        )
        sol2 = {("A", "4"), ("B", "2"), ("C", "5"), ("D", "6"), ("E", "3"), ("D", "1")}
        self.assertEqual(
            solver.find_violation(sol2, 1, True), "not_enough_lights_in_night"
        )
        sol3 = {("A", "1"), ("B", "2"), ("C", "3"), ("D", "4"), ("E", "5"), ("E", "6")}
        self.assertIsNone(solver.find_violation(sol3, 2, True))

    def test_check_multiple_match_logic(self):
        solver = Solver(toy_season)
        sol1 = {
            ("A", "1"),
            ("B", "2"),
            ("C", "3"),
            ("D", "4"),
            ("E", "5"),
            ("E", "6"),
            ("D", "7"),
        }
        self.assertEqual(
            solver.find_multiple_match_violation(sol1, 1), "too_many_double_matches"
        )
        sol2 = {("A", "1"), ("B", "2"), ("C", "3"), ("D", "4"), ("D", "5"), ("E", "6")}
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 1),
            "mm_not_in_complete_multiple_match",
        )
        sol3 = {
            ("A", "1"),
            ("A", "2"),
            ("B", "1"),
            ("B", "2"),
        }
        self.assertEqual(
            solver.find_multiple_match_violation(sol3, 1),
            "too_many_double_matches",
        )
        sol4 = {("A", "1"), ("A", "2"), ("A", "3")}
        self.assertEqual(
            solver.find_multiple_match_violation(sol4, 1),
            "triple_match_not_allowed",
        )

    def test_solve_with_dm(self):
        solver = Solver(toy_season)
        self.compare_solve_approaches(solver, end=2)

    def test_solve_with_dm2(self):
        solver = Solver(toy_season2)
        self.compare_solve_approaches(solver, end=4)

    def test_check_multiple_match_logic_wo_dm(self):
        # season without dm specified
        solver = Solver(toy_season_wo_dm)
        sol1 = {("A", "1"), ("B", "2"), ("C", "3"), ("D", "4"), ("D", "5"), ("E", "6")}
        self.assertFalse(solver.find_multiple_match_violation(sol1, 2))
        sol2 = {
            ("A", "1"),
            ("A", "2"),
            ("B", "1"),
            ("B", "2"),
        }
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 1),
            "too_many_double_matches",
        )
        sol3 = {("A", "1"), ("A", "2"), ("A", "3")}
        self.assertEqual(
            solver.find_multiple_match_violation(sol3, 1),
            "triple_match_not_allowed",
        )

    def test_solve_without_dm(self):
        solver = Solver(toy_season_wo_dm)
        self.compare_solve_approaches(solver)

    def test_check_multiple_match_logic_with_double_match_pair(self):
        solver = VIP2023Solver(toy_season_with_double_match_pair)
        sol1 = {("A", "1"), ("B", "2"), ("C", "3"), ("D", "4"), ("D", "5"), ("A", "6")}
        self.assertTrue(solver.find_multiple_match_violation(sol1, 3))
        sol2 = {
            ("A", "1"),
            ("B", "6"),
        }
        self.assertEqual(
            solver.find_multiple_match_violation(sol2, 3),
            "vip23_double_match_pair_not_same_match",
        )
        sol3 = {("A", "1"), ("A", "2")}
        self.assertEqual(
            solver.find_multiple_match_violation(sol3, 3),
            "double_match_lefts_not_double_match_pair",
        )
        sol4 = {("A", "1"), ("A", "2"), ("A", "3")}
        self.assertEqual(
            solver.find_multiple_match_violation(sol4, 1),
            "triple_match_not_allowed",
        )

    def compare_solve_approaches(self, solver: Solver, end: int = 5):
        season = solver.season
        empty_season = Season(
            "empty",
            season.lefts,
            season.rights,
            (),
            Matchboxes(),
            mm=season.mm,
            num_matches=season.num_matches,
            double_match_pair=season.double_match_pair,
            double_match_pair_known=season.double_match_pair_known,
            mm_known_after_week=season.mm_known_after_week,
            max_mm_size=season.max_mm_size,
            mb_reveals_dm=season.mb_reveals_dm,
        )
        empty_season = Solver(empty_season)
        sols = set(empty_season.generate_complete_solutions(frozenset(), end, True))
        sols = [s for s in sols if solver.solution_possible(s, end, True)]
        self.assertEqual(len(sols), len(solver.solve(end)))
