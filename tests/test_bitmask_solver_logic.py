import functools
import unittest
from typing import override

import tests._test_bases as base
from ayto import BitMaskSolver, Season, get_solver
from tests.fixtures import BITMASK_SOLVERS


@functools.cache
def _real_solver(name: str) -> BitMaskSolver:
    return get_solver(name, True)


class TestBitMaskSolverLogic(base.SolverLogicTestsBase[int]):
    @override
    def make_solver(self, season: Season) -> BitMaskSolver:
        return BITMASK_SOLVERS[season.match_model](season)

    @override
    def real_solver(self, name: str) -> BitMaskSolver:
        return _real_solver(name)


if __name__ == "__main__":
    unittest.main()  # pyright: ignore[reportUnusedCallResult]
