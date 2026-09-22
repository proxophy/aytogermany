import unittest
from ayto.utils import get_solver


class TestSeasonSolution(unittest.TestCase):
    def test_normalo2020(self):
        self.check_season("normalo2020")

    def test_normalo2021(self):
        self.check_season("normalo2021")

    def test_normalo2022(self):
        self.check_season("normalo2022")

    def test_normalo2023(self):
        self.check_season("normalo2023")

    def test_normalo2024(self):
        self.check_season("normalo2024")

    def test_normalo2025(self):
        self.check_season("normalo2025")

    def test_normalo2026(self):
        self.check_season("normalo2026")

    def test_vip2021(self):
        self.check_season("vip2021")

    def test_vip2022(self):
        self.check_season("vip2022")

    def test_vip2023(self):
        self.check_season("vip2023")

    def test_vip2024(self):
        self.check_season("vip2024")

    def test_vip2025(self):
        self.check_season("vip2025")

    def test_vip2026(self):
        self.check_season("vip2026")

    def check_season(self, sn):
        solver = get_solver(sn)
        solution = solver.season.solution
        if solution:
            solpos = solver.solution_possible(solution, 10, True)
            self.assertTrue(solpos["res"], solpos["reason"])
