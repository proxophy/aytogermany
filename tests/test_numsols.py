import unittest

import pandas as pd

from ayto.utils import get_solver

df = pd.read_csv("analytics/num_solutions_extended.csv", index_col=0)
df.index = df.index.str.strip()
df.columns = df.columns.astype(str).str.strip()

class TestSeasonNumSols(unittest.TestCase):
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

    def check_season(self, sn: str, start: int = 2):

        for i in range(start, 10):
            with self.subTest(episode=i):
                solver = get_solver(sn)
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


if __name__ == "__main__":
    _ = unittest.main()
