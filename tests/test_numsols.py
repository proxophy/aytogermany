import functools
import unittest
from typing import Any, override

import tests._test_bases as base
from ayto import (
    Season,
    Solution,
    get_solver,
)
from ayto.base import SolutionType, Solver
from ayto.bitmask_solver import BitMaskSolver
from ayto.tuple_solver import TupleSolver
from tests.fixtures import BITMASK_SOLVERS, TUPLE_SOLVERS

type AnySolver[S: SolutionType] = Solver[S, Any]  # pyright: ignore[reportExplicitAny]


@functools.cache
def _real_tuple_solver(name: str) -> TupleSolver:
    return get_solver(name)


class TestTupleSolverNumSols(base.NumSolTestBase[Solution]):
    @override
    def make_solver(self, season: Season) -> TupleSolver:
        return TUPLE_SOLVERS[season.match_model](season)

    @override
    def real_solver(self, name: str) -> TupleSolver:
        return _real_tuple_solver(name)


@functools.cache
def _real_bitmask_solver(name: str) -> BitMaskSolver:
    return get_solver(name, True)


class TestBitMaskNumSols(base.NumSolTestBase[int]):
    @override
    def make_solver(self, season: Season) -> BitMaskSolver:
        return BITMASK_SOLVERS[season.match_model](season)

    @override
    def real_solver(self, name: str) -> BitMaskSolver:
        return _real_bitmask_solver(name)


if __name__ == "__main__":
    unittest.main()  # pyright: ignore[reportUnusedCallResult]
