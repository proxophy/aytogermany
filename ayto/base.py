import functools
import itertools
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable
from typing import Any, Protocol, Self

# from line_profiler import profile
from ayto.season import Knowledge, Pair, Person, Season


class SolutionType(Hashable, Protocol):
    def __or__(self, other: Self, /) -> Self: ...
    def __and__(self, other: Self, /) -> Self: ...


class Solver[SolType: SolutionType](ABC):
    """SolutionType: how a (partial) solution is stored: frozenset[Pair] or an int mask."""

    def __init__(self, season: Season) -> None:
        super().__init__()
        self.season: Season = season
        self._no_match_cache: dict[tuple[Person, Person, int, bool], bool] = {}
        self.mm_person_cache: list[Person | None] = [
            season.get_mm(ep) for ep in range(self.season.max_lights + 1)
        ]
        self.knowledge_cache: dict[tuple[int, bool], Knowledge] = {}

    @abstractmethod
    def from_pairs(self, pairs: Iterable[Pair]) -> SolType:
        pass

    @abstractmethod
    def no_match_reason(
        self, l: int, r: int, episode: int, include_night: bool
    ) -> str | None:
        """Return the reason why `(l, r)` is known not to be a match up to a given episode"""

    def str_no_match_reason(
        self, l: str, r: str, episode: int, include_night: bool
    ) -> str | None:
        li = self.season.lefts_names.index(l)
        ri = self.season.rights_names.index(r)
        return self.no_match_reason(li, ri, episode, include_night)

    def no_match(self, l: int, r: int, end: int, include_night: bool) -> bool:
        """Return whether `(l, r)` is known not to be a match up to a given episode"""
        key = (l, r, end, include_night)
        cached = self._no_match_cache.get(key)
        if cached is None:
            cached = self.no_match_reason(l, r, end, include_night) is not None
            self._no_match_cache[key] = cached
        return cached

    @abstractmethod
    def find_format_violation(self, sol: SolType) -> str | None:
        """Return the first format violation found in the solution, if any"""

    @abstractmethod
    def find_multiple_match_violation(self, sol: SolType, episode: int) -> str | None:
        """Return the first multiple-match violation found in the solution, if any"""

    @abstractmethod
    def find_violation(
        self, sol: SolType, episode: int, include_night: bool
    ) -> str | None:
        """Return the first logic violation found in the solution, if any"""

    def solution_possible(
        self, sol: SolType, episode: int, include_night: bool
    ) -> bool:
        """Check whether the solution is feasible up to a given episode

        Args:
            sol: Solution containing the pairs to check
            episode: Last episode to consider
            include_night: Whether the matching night of the last episode should be considered

        Returns:
            `True` if no violation was found, ``False`` otherwise.
        """
        return self.find_violation(sol, episode, include_night) is None

    def combine_possible_solutions(
        self,
        solutions1: list[SolType],
        solutions2: list[SolType],
        episode: int,
        include_night: bool,
    ) -> list[SolType]:
        """Combines solutions from two lists and keeps feasible combinations, given `episode`(inclusive) and `include_night`"""

        m_asm: set[SolType] = set()
        for s1, s2 in itertools.product(solutions1, solutions2):
            s3 = s1 | s2
            if self.solution_possible(s3, episode, include_night):
                m_asm.add(s3)

        return list(m_asm)

    def generate_partial_solutions(
        self, episode: int, include_night: bool
    ) -> list[SolType]:
        """Generate partial solutions from the matching night.

        For each matching night, the required number of matching pairs is selected and the resulting partial solutions are combined.
        Args:
            episode (int): Last episode to consider, inclusive.
            include_night (bool): Whether to consider the matching night of
                the last episode.

        Returns:
            list[Solution]: Partial solutions consistent with the matching-night
                results and known constraints.
        """

        knowledge = self.knowledge_cache[(episode, include_night)]
        nights = knowledge.nights
        kpm = knowledge.perfect_matches

        if len(nights) == 0:
            return [self.from_pairs(kpm)]

        pairs_per_night: list[list[SolType]] = []
        for i, night in enumerate(nights):
            invalid_pairs = list(
                filter(
                    lambda p: self.no_match(*p, episode, include_night) or p in kpm,
                    night.pairs,
                )
            )
            definite_matches = list(filter(lambda p: p in kpm, night.pairs))
            remaining = set(night.pairs) - set(definite_matches) - set(invalid_pairs)

            missing_lights = night.lights - len(definite_matches)
            if missing_lights < 0:
                raise ValueError(
                    f"{self.season.name}, night {i}: {len(definite_matches)} known perfect  \
                    matches are seated but the night only has {night.lights} lights"
                )
            combinations = [
                self.from_pairs(set(comb) | set(kpm))
                for comb in itertools.combinations(remaining, missing_lights)
            ]

            pairs_per_night.append(combinations)

        merged_solutions: list[SolType] = functools.reduce(
            lambda s1, s2: self.combine_possible_solutions(
                s1, s2, episode, include_night
            ),
            pairs_per_night,
        )

        return merged_solutions

    @abstractmethod
    def possible_matches_for_solution(
        self, sol: SolType, episode: int, include_night: bool
    ) -> dict[int, Any]:  # pyright: ignore[reportExplicitAny]
        """Given a solution, return a dict for the possible matches for the lefts that are not in the solution, given `episode`(inclusive) and `include_night`"""

    @abstractmethod
    def generate_complete_solutions(
        self, sol: SolType, episode: int, include_night: bool
    ) -> list[SolType]:
        """Generate complete solutions by extending a partial solution.

        Args:
            sol: Partial solution to extend.
            episode (int): Last episode to consider, inclusive.
            include_night (bool): Whether to consider the matching night of
                the last episode.

        Returns:
            list: Complete solutions consistent with the given partial
                solution and the known constraints.
        """

    @abstractmethod
    def merge_mm_in_solution(
        self,
        sol: SolType,
        partial_solutions: list[tuple[SolType, Any]],  # pyright: ignore[reportExplicitAny]
    ) -> list[SolType]:
        """Merge solution with partial solutions to complete solution when the multiple match(es) are present the solution, given `episode`(inclusive)"""

    @abstractmethod
    def merge_mm_not_in_solution(
        self,
        sol: SolType,
        partial_solutions: list[tuple[SolType, Any]],  # pyright: ignore[reportExplicitAny]
        episode: int,
        include_night: bool,
    ) -> list[SolType]:
        """Merge solution with partial solutions to complete solutions when the multiple match(es) are not present the solution, given `episode`(inclusive) and `include_night`"""

    def solve(
        self, episode: int, include_night: bool = True, validate: bool = False
    ) -> list[SolType]:
        """Solve season and return all possible solutions

        Args:
            episode (int): Last episode to consider
            include_night (bool): Whether the matching night of the last episode should be considered.Defaults to True.
            validate (bool, optional): Whether solutions should be validated by solution_possible after generating them. Defaults to False.

        Returns:
            list: list of all possible solutions
        """

        # reduction of the search space by generating partial solutions from nights
        partial_solutions = self.generate_partial_solutions(episode, include_night)

        solutions_unfiltered: list[SolType] = []
        for g in partial_solutions:
            sols_g = self.generate_complete_solutions(g, episode, include_night)
            solutions_unfiltered.extend(sols_g)

        # the tests should guarantee that all generated solutions are possible
        if validate:
            solutions = [
                s
                for s in solutions_unfiltered
                if self.solution_possible(s, episode, include_night)
            ]
        else:
            solutions = list(solutions_unfiltered)

        return solutions
