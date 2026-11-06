import unittest

import pandas as pd

from ayto import get_season
from ayto.bitmask_solver import BitMaskSolver
from ayto.utils import get_solver

df = pd.read_csv("analytics/num_solutions_extended.csv", index_col=0)
# df.columns = df.columns.astype(int)
df.index = df.index.str.strip()
df.columns = df.columns.astype(str).str.strip()


class TestSeasonNumSols(unittest.TestCase):
    def test_normalo2020_extended(self):
        self.check_season_extended("normalo2020")

    def test_normalo2021_extended(self):
        self.check_season_extended("normalo2021")

    def test_normalo2022_extended(self):
        self.check_season_extended("normalo2022")

    def test_normalo2023_extended(self):
        self.check_season_extended("normalo2023")

    # def test_normalo2024_extended(self):
    #     self.check_season_extended("normalo2024")

    def test_normalo2025_extended(self):
        self.check_season_extended("normalo2025")

    def test_normalo2026_extended(self):
        self.check_season_extended("normalo2026")

    def test_vip2021_extended(self):
        self.check_season_extended("vip2021")

    def test_vip2022_extended(self):
        self.check_season_extended("vip2022")

    # def test_vip2023_extended(self):
    #     self.check_season_extended("vip2023")

    def test_vip2024_extended(self):
        self.check_season_extended("vip2024")

    # def test_vip2025_extended(self):
    #     self.check_season_extended("vip2025", start=3)

    # def test_vip2026_extended(self):
    #     self.check_season_extended("vip2026")

    def check_season_extended(self, sn: str, start: int = 2):
        for i in range(start, 10):
            with self.subTest(episode=i, include_night=False):
                season = get_season(sn)
                solver = BitMaskSolver(season)
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
