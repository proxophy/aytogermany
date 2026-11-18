"""Shared toy seasons and solver registries for the solver tests."""

from ayto import Matchboxes, Night, Season
from ayto.bitmask_solver import BitMaskSolver
from ayto.double_match_for_right_solver import (
    DoubleMatchForRightBitMaskSolver,
    DoubleMatchForRightSolver,
)
from ayto.double_match_pair_solver import (
    DoubleMatchPairBitMaskSolver,
    DoubleMatchPairSolver,
)
from ayto.triple_match_solver import TripleMatchBitMaskSolver, TripleMatchSolver
from ayto.tuple_solver import TupleSolver
from ayto.two_double_matches_solver import (
    TwoDoubleMatchesBitMaskSolver,
    TwoDoubleMatchesSolver,
)

# match_model -> solver class (the tuple and the bitmask family)
TUPLE_SOLVERS: dict[str, type[TupleSolver]] = {
    "double_match": TupleSolver,
    "double_match_pair": DoubleMatchPairSolver,
    "triple_match": TripleMatchSolver,
    "two_double_matches": TwoDoubleMatchesSolver,
    "double_match_for_right": DoubleMatchForRightSolver,
}
BITMASK_SOLVERS: dict[str, type[BitMaskSolver]] = {
    "double_match": BitMaskSolver,
    "double_match_pair": DoubleMatchPairBitMaskSolver,
    "triple_match": TripleMatchBitMaskSolver,
    "two_double_matches": TwoDoubleMatchesBitMaskSolver,
    "double_match_for_right": DoubleMatchForRightBitMaskSolver,
}

toy_nights = (
    Night({(0, 0), (1, 2), (2, 3), (3, 4), (4, 1)}, 1),
    Night({(0, 4), (1, 1), (2, 2), (3, 3), (4, 0)}, 3),
    Night({(0, 4), (1, 1), (2, 2), (3, 3), (4, 5)}, 4),
)
toy_matchboxes = Matchboxes((0,), ((1, 2),), (False,), (True,))
toy_lefts = ("A", "B", "C", "D", "E")
toy_rights = ("1", "2", "3", "4", "5", "6")
toy_season = Season(
    "toy", toy_lefts, toy_rights, toy_nights, toy_matchboxes, mm_person="6"
)

toy_season2 = Season(
    name="toy",
    lefts_names=("A", "B", "C", "D", "E"),
    rights_names=("1", "2", "3", "4", "5", "6"),
    nights=(
        Night(pairs={(4, 3), (0, 2), (1, 1), (3, 0), (2, 4)}, lights=1),
        Night(pairs={(0, 5), (1, 2), (4, 1), (3, 0), (2, 4)}, lights=1),
        Night(pairs={(2, 1), (1, 2), (4, 3), (3, 0), (0, 4)}, lights=0),
        Night(pairs={(3, 1), (1, 0), (4, 3), (2, 2), (0, 4)}, lights=1),
        Night(pairs={(4, 2), (2, 0), (1, 1), (3, 3), (0, 4)}, lights=2),
    ),
    matchboxes=Matchboxes(
        weeks=(0, 1, 3, 4),
        pairs=((0, 1), (3, 1), (3, 0), (2, 4)),
        results=(False, False, False, False),
        before_night=(True, True, True, True),
    ),
    solution=frozenset({(0, 0), (0, 5), (4, 4), (2, 2), (1, 1), (3, 3)}),
    mm_person="6",
    double_match_pair_known=7,
    mm_known_from_week=0,
    matchbox_reveals_dm=True,
)

toy_season_wo_dm = Season(
    "toy_wo_dm", toy_lefts, toy_rights, toy_nights, toy_matchboxes
)
toy_season_dm_later_known = Season(
    "toy_dm_later_known",
    toy_lefts,
    toy_rights,
    toy_nights,
    toy_matchboxes,
    mm_person="6",
    mm_known_from_week=2,
)

toy_season_with_double_match_pair = Season(
    "toy_with_dm_pair",
    toy_lefts,
    toy_rights,
    toy_nights,
    Matchboxes((0,), ((2, 0),), (False,), (True,)),
    match_model="double_match_pair",
    mm_person="6",
    double_match_pair_names=("1", "6"),
    double_match_pair_known=0,
)

toy_season_with_triple_match = Season(
    "toy",
    toy_lefts,
    ("1", "2", "3", "4", "5", "6", "7"),
    toy_nights,
    toy_matchboxes,
    match_model="triple_match",
    mm_person="6",
)

toy_season_with_two_double_matches = Season(
    "toy",
    toy_lefts,
    ("1", "2", "3", "4", "5", "6", "7"),
    toy_nights,
    toy_matchboxes,
    match_model="two_double_matches",
    mm_person="6",
)

toy_season_with_dm_for_right = Season(
    "toy",
    ("A", "B", "C", "D", "E", "F"),
    ("1", "2", "3", "4", "5", "6"),
    toy_nights,
    toy_matchboxes,
    match_model="double_match_for_right",
    extra_left_name="F",
)

model_to_toy = {
    "double_match": toy_season,
    "double_match_pair": toy_season_with_double_match_pair,
    "two_double_matches": toy_season_with_two_double_matches,
    "double_match_for_right": toy_season_with_dm_for_right,
    "triple_match": toy_season_with_triple_match,
}

# every toy season, by a unique label (used by the backend-agreement tests)
ALL_TOY_SEASONS = {
    "double_match": toy_season,
    "double_match_2": toy_season2,
    "wo_dm": toy_season_wo_dm,
    "dm_later_known": toy_season_dm_later_known,
    "double_match_pair": toy_season_with_double_match_pair,
    "triple_match": toy_season_with_triple_match,
    "two_double_matches": toy_season_with_two_double_matches,
    "double_match_for_right": toy_season_with_dm_for_right,
}
