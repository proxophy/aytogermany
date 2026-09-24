import pandas as pd
import unittest

# from ayto import *
from ayto.ayto import Solver
from ayto.utils import get_solver
from ayto.models import Season

df = pd.read_csv("analytics/num_solutions.csv", index_col=0, header=0)
df.columns = df.columns.astype(int)



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

    def check_season(self, sn, start=2):
        row = df.loc[sn]

        for i in range(start, 10):
            with self.subTest(episode=i):
                solver = get_solver(sn)
                sols = solver.solve(i, True)

                self.assertEqual(
                    len(sols),
                    row[i],
                    f"{sn} i={i}: got {len(sols)}, expected {row[i]}",
                )


if __name__ == "__main__":
    allseasons = [
        "normalo2020",
        "normalo2021",
        "normalo2022",
        "normalo2023",
        # "normalo2024",
        "normalo2025",
        "normalo2026",
        "vip2021",
        "vip2022",
        "vip2023",
        "vip2024",
        # "vip2025",
        "vip2026",
    ]
    unittest.main()
