from typing import override

from ayto.season import Pair

from .season import *
from .solver import Solver


class DoubleMatchPairSolver(Solver):
    @override
    def no_match_reason(self, l: Person, r: Person, episode: int, include_night : bool) -> str | None:
        assert self.season.double_match_pair
        knowledge = self.knowledge_cache[(episode, include_night)]
        if episode >= self.season.double_match_pair_known:
            no_matches: list[Pair] = knowledge.mb_no_matches

            if r in self.season.double_match_pair:
                for dmr in self.season.double_match_pair:
                    if (l, dmr) in no_matches:
                        return "matchbox_false_for_double_match_pair"
        return super().no_match_reason(l, r, episode, include_night)

    @override
    def find_multiple_match_violation(
        self, sol: SolutionType, episode: int
    ) -> str | None:
        assert self.season.double_match_pair

        if episode >= self.season.double_match_pair_known:
            # lefts that partner with double_match_pair
            dml = [l for (l, r) in sol if r in self.season.double_match_pair]
            if len(dml) > 1 and dml[0] != dml[1]:
                return "double_match_pair_not_same_match"

            sol_dict = dict_rep(sol)

            # lefts with multiple partners
            mult_lefts = [l for l, rs in sol_dict.items() if len(rs) > 1]
            if len(mult_lefts) > 0:
                mult_left = mult_lefts[0]
                mult_rights = sol_dict[mult_left]
                if len(mult_rights) == 2 and set(self.season.double_match_pair) != set(
                    mult_rights
                ):
                    return "double_match_lefts_not_double_match_pair"

        return super().find_multiple_match_violation(sol, episode)

    @override
    def merge_mm_not_in_solution(
        self,
        sol: Solution,
        partial_solutions: list[Solution],
        episode: int,
        include_night: bool,
    ) -> list[Solution]:
        assert self.season.double_match_pair

        if episode < self.season.double_match_pair_known:
            return super().merge_mm_not_in_solution(
                sol, partial_solutions, episode, include_night
            )

        solutions: list[Solution] = []
        knowledge = self.knowledge_cache[(episode, include_night)]
        sitting_nomatches = knowledge.get_sitting_no_matches(sol)


        for partial_solution in partial_solutions:
            ten_matches = sol.union(partial_solution)

            _, sitting_rights = zip(*ten_matches)
            # right that is missing in solution
            mr = next(r for r in self.season.rights if r not in sitting_rights)

            if mr not in self.season.double_match_pair:
                continue
            # find left that pairs with one person of double_match_pair
            dmleft = next(
                l for (l, r) in ten_matches if  r in self.season.double_match_pair
            )
            if (dmleft, mr) in sitting_nomatches:
                continue
            solutions.append(ten_matches | {(dmleft, mr)})

        return solutions
