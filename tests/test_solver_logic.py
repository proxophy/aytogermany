import functools
import unittest
from typing import override

import tests._test_bases as base
from ayto import Season, Solution, get_solver
from ayto.tuple_solver import TupleSolver
from tests.fixtures import TUPLE_SOLVERS


@functools.cache
def _real_tuple_solver(name: str) -> TupleSolver:
    return get_solver(name)


class TestTupleSolverLogic(base.SolverLogicTestsBase[Solution]):
    @override
    def make_solver(self, season: Season) -> TupleSolver:
        return TUPLE_SOLVERS[season.match_model](season)

    @override
    def real_solver(self, name: str) -> TupleSolver:
        return _real_tuple_solver(name)


if __name__ == "__main__":
    unittest.main()  # pyright: ignore[reportUnusedCallResult]
