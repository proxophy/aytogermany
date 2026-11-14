import unittest

import pandas as pd

from ayto import get_season
from ayto.bitmask_solver import BitMaskSolver
from ayto.tuple_solver import TupleSolver
from ayto.utils import get_solver

df = pd.read_csv("analytics/num_solutions_extended.csv", index_col=0)
# df.columns = df.columns.astype(int)
df.index = df.index.str.strip()
df.columns = df.columns.astype(str).str.strip()


class TestSeasonNumSols(unittest.TestCase):
    def test_normalo2020_numsols(self):
        self.check_season_numsols("normalo2020")

    def test_normalo2021_numsols(self):
        self.check_season_numsols("normalo2021")

    def test_normalo2022_numsols(self):
        self.check_season_numsols("normalo2022")

    def test_normalo2023_numsols(self):
        self.check_season_numsols("normalo2023")

    def test_normalo2024_numsols(self):
        self.check_season_numsols("normalo2024")

    def test_normalo2025_numsols(self):
        self.check_season_numsols("normalo2025")

    def test_normalo2026_numsols(self):
        self.check_season_numsols("normalo2026")

    def test_vip2021_numsols(self):
        self.check_season_numsols("vip2021")

    def test_vip2022_numsols(self):
        self.check_season_numsols("vip2022")

    def test_vip2023_numsols(self):
        self.check_season_numsols("vip2023")

    def test_vip2024_numsols(self):
        self.check_season_numsols("vip2024")

    def test_vip2025_numsols(self):
        self.check_season_numsols("vip2025", start=3)

    def test_vip2026_numsols(self):
        self.check_season_numsols("vip2026")

    def test_normalo2020_nomatches(self):
        self.compare_no_matches("normalo2020")
    
    def test_normalo2021_nomatches(self):
        self.compare_no_matches("normalo2021")

    def test_normalo2022_nomatches(self):
        self.compare_no_matches("normalo2022")

    def test_normalo2023_nomatches(self):
        self.compare_no_matches("normalo2023")

    def test_normalo2024_nomatches(self):
        self.compare_no_matches("normalo2024")

    def test_normalo2025_nomatches(self):
        self.compare_no_matches("normalo2025")

    def test_normalo2026_nomatches(self):
        self.compare_no_matches("normalo2026")

    def test_vip2021_nomatches(self):
        self.compare_no_matches("vip2021")

    def test_vip2022_nomatches(self):
        self.compare_no_matches("vip2022")

    def test_vip2023_nomatches(self):
        self.compare_no_matches("vip2023")

    def test_vip2024_nomatches(self):
        self.compare_no_matches("vip2024")

    def test_vip2025_nomatches(self):
        self.compare_no_matches("vip2025")

    def test_vip2026_nomatches(self):
        self.compare_no_matches("vip2026")

    def check_season_numsols(self, sn: str, start: int = 2):
        
        solver: BitMaskSolver = get_solver(sn, True)
        print(solver.__class__)
        self.assertIsInstance(solver, BitMaskSolver)
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
    def compare_no_matches(self, sn: str):
        bmsolver: BitMaskSolver = get_solver(sn, True)
        tsolver: TupleSolver = get_solver(sn, False)
        season = get_season(sn)
        for i in range(season.num_weeks):
            for b in [False, True]:
                with self.subTest(episode=i, include_night = b):
                    for l in season.lefts:
                        for r in season.rights:
                            self.assertEqual(bmsolver.no_match_reason(l, r, i, b),
                            tsolver.no_match_reason(l, r, i, b))



if __name__ == "__main__":
    _ = unittest.main()
