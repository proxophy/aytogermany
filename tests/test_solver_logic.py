import unittest

from ayto.utils import get_solver
from ayto import Solver
from ayto.models import Season, Night, Matchboxes

normalo2020 = get_solver("normalo2020")
normalo2024 = get_solver("normalo2024")
normalo2026 = get_solver("normalo2026")
vip2023 = get_solver("vip2023")

toy_nights = (
    Night(
        set(
            (
                ("A", "1"),
                ("B", "3"),
                ("C", "4"),
                ("D", "5"),
                ("E", "2"),
            )
        ),
        1,
    ),
    Night(
        set(
            (
                ("A", "5"),
                ("B", "2"),
                ("C", "3"),
                ("D", "4"),
                ("E", "1"),
            )
        ),
        3,
    ),
     Night(
            set(
                (
                    ("A", "5"),
                    ("B", "2"),
                    ("C", "3"),
                    ("D", "4"),
                    ("E", "6"),
                )
            ),
            4,
        )
)
toy_matchboxes = Matchboxes((0,), ((("B", "3")),), (False,))
toy_lefts = ("A", "B", "C", "D", "E")
toy_rights = ("1", "2", "3", "4", "5", "6")
toy_season = Season(
    "toy", toy_lefts, toy_rights, toy_nights, toy_matchboxes, mm="6", num_matches=6
)


class TestSeasonSolution(unittest.TestCase):
    def test_no_match_matchbox(self):
        self.assertTrue(normalo2020.no_match("Ivana", "Mo", 2))

    def test_no_match_matchbox_end(self):
        self.assertFalse(normalo2020.no_match("Luisa", "Ferhat", 2))
        self.assertTrue(normalo2020.no_match("Luisa", "Ferhat", 10))

    def test_no_match_bo_night(self):
        self.assertTrue(normalo2020.no_match("Nadine", "Dominic", 10))

    def test_no_match_bo_night_end(self):
        self.assertFalse(normalo2020.no_match("Madleine", "Ferhat", 2))
        self.assertTrue(normalo2020.no_match("Madleine", "Ferhat", 3))

    def test_no_match_double_match(self):
        self.assertTrue(normalo2020.no_match("Sabrina", "Mo", 2))
        self.assertTrue(normalo2020.no_match("Aline", "Ferhat", 2))
        self.assertFalse(normalo2020.no_match("Aline", "Mo", 2))

    def test_no_match_double_match_pair(self):
        self.assertTrue(vip2023.no_match("Sabrina", "Max", 7))
        self.assertTrue(vip2023.no_match("Sabrina", "Peter", 7))

    def test_no_match_tripple_match(self):
        self.assertFalse(normalo2024.no_match("Gerrit", "Edda", 3))
        self.assertFalse(normalo2024.no_match("Gerrit", "Pia", 3))
        self.assertTrue(normalo2024.no_match("Gerrit", "Edda", 4))

    def test_no_match_dm_later_known(self):
        self.assertFalse(normalo2026.no_match("Noel", "Alicia", 5))
        self.assertFalse(normalo2026.no_match("Noel", "Alicia", 6))
        self.assertTrue(normalo2026.no_match("Julian M", "Linda", 6))
        self.assertTrue(normalo2026.no_match("Julian M", "Alicia", 6))
        self.assertTrue(normalo2026.no_match("Julian M", "Alicia", 4))

    def test_solution_correct_format(self):
        toy_solver = Solver(toy_season)
        sol1 = {("A", "1"), ("B", "2"), ("C", "3"), ("D", "4"), ("E", "5"), ("E", "6")}

        self.assertTrue(toy_solver.solution_has_correct_format(sol1)["res"])
        sol1.add(("B", "3"))
        self.assertFalse(toy_solver.solution_has_correct_format(sol1)["res"])
        self.assertFalse(
            toy_solver.solution_has_correct_format({("A", "1"), ("B", "1")})["res"]
        )
        self.assertFalse(
            toy_solver.solution_has_correct_format({("A", "1"), ("B", "2s")})["res"]
        )
        self.assertFalse(
            toy_solver.solution_has_correct_format({("A", "1"), ("Bs", "2")})["res"]
        )

    def test_solution_possible_no_match(self):
        toy_solver = Solver(toy_season)
        sol1 = {("A", "1"), ("B", "3"), ("C", "2"), ("D", "4"), ("E", "5"), ("E", "6")}
        self.assertDictEqual(
            toy_solver.solution_possible(sol1, 1, True),
            {
                "res": False,
                "reason": "known_no_match",
                "detail": ("B", "3"),
            },
        )

    def test_solution_possible_lights(self):
        toy_solver = Solver(toy_season)
        sol1 = {("A", "1"), ("B", "2"), ("C", "4"), ("D", "5"), ("E", "3"), ("E", "6")}
        self.assertDictEqual(
            toy_solver.solution_possible(sol1, 1, True),
            {"res": False, "reason": "too_many_lights_in_night", "detail": 0},
        )
        sol2 = {("A", "4"), ("B", "2"), ("C", "5"), ("D", "6"), ("E", "3"), ("D", "1")}
        self.assertDictEqual(
            toy_solver.solution_possible(sol2, 1, True),
            {"res": False, "reason": "not_enough_lights_in_night", "detail": 0},
        )
        sol3 = {("A", "1"), ("B", "2"), ("C", "3"), ("D", "4"), ("E", "5"), ("E", "6")}
        self.assertTrue(toy_solver.solution_possible(sol3, 2, True)["res"])

    def test_check_multiple_match_logic(self):
        toy_solver = Solver(toy_season)
        sol1 = {
            ("A", "1"),
            ("B", "2"),
            ("C", "3"),
            ("D", "4"),
            ("E", "5"),
            ("E", "6"),
            ("D", "7"),
        }
        self.assertFalse(toy_solver.check_multiple_match_logic(sol1, 1)["res"])
        sol1 = {("A", "1"), ("B", "2"), ("C", "3"), ("D", "4"), ("D", "5"), ("E", "6")}
        self.assertEqual(
            toy_solver.check_multiple_match_logic(sol1, 1)["reason"],
            "mm_not_in_complete_multiple_match",
        )
        sol1 = {
            ("A", "1"),
            ("A", "2"),
            ("B", "1"),
            ("B", "2"),
        }
        self.assertEqual(
            toy_solver.check_multiple_match_logic(sol1, 1)["reason"],
            "too_many_double_matches",
        )
        sol1 = {("A", "1"), ("A", "2"), ("A", "3")}
        self.assertEqual(
            toy_solver.check_multiple_match_logic(sol1, 1)["reason"],
            "triple_match_not_allowed",
        )

    def test_solve_withdm(self):
        toy_solver = Solver(toy_season)
        self.compare_solve_approaches(toy_solver)

    def compare_solve_approaches(self, solver: Solver):
        end = 5
        sols = solver.generate_complete_solutions(frozenset(), end, True)
        sols = [s for s in sols if solver.solution_possible(s, end, True)["res"]]
        self.assertEqual(len(sols), len(solver.solve(end)))
