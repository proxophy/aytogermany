import functools

from .ayto import Solver, mm_left, dict_rep, get_candidates
from .models import *


class VIP2023Solver(Solver):
    @functools.cache
    def no_match(self, l: str, r: str, end: int) -> bool:
        assert self.season.double_match_pair
        if end >= self.season.double_match_pair_known:
            mb = self.season.get_matchboxes(end)

            if r in self.season.double_match_pair:
                for r in self.season.double_match_pair:
                    if (l, r) in mb and not mb[(l, r)]:
                        return True
        return super().no_match(l, r, end)

    def check_multiple_match_logic(self, sol: Solution | set[Pair], end: int) -> dict:
        assert self.season.double_match_pair

        if end >= self.season.double_match_pair_known:
            # lefts that partner with double_match_pair
            dml = [l for (l, r) in sol if r in self.season.double_match_pair]
            if len(dml) > 1 and dml[0] != dml[1]:
                return {
                    "res": False,
                    "reason": "vip23_double_match_pair_not_same_match",
                }

            sol_dict = dict_rep(sol)

            # lefts with multiple partners
            mult_lefts = [l for l, rs in sol_dict.items() if len(rs) > 1]
            if len(mult_lefts) > 0:
                mult_left = mult_lefts[0]
                mult_rights = sol_dict[mult_left]
                if len(mult_rights) == 2 and set(self.season.double_match_pair) != set(
                    mult_rights
                ):
                    return {
                        "res": False,
                        "reason": "double_match_lefts_not_double_match_pair",
                    }

        return super().check_multiple_match_logic(sol, end)

    def merge_mm_not_in_solution(
        self,
        sol: Solution,
        partial_solutions: list[Solution],
        end: int,
        include_night: bool,
    ) -> list[Solution]:
        assert self.season.double_match_pair

        if end < self.season.double_match_pair_known:
            return super().merge_mm_not_in_solution(sol, partial_solutions, end, include_night)

        solutions = []
        for partial_solution in partial_solutions:
            ten_matches = sol.union(partial_solution)

            _, sitting_rights = zip(*ten_matches)
            # right that is missing in solution
            mr = [r for r in self.season.rights if r not in sitting_rights][0]
            
            if mr not in self.season.double_match_pair:
                continue
            # find left that pairs with one person of double_match_pair
            dmleft = [
                l for (l, r) in ten_matches if r in self.season.double_match_pair
            ][0]
            solutions.append(ten_matches | {(dmleft, mr)})

        return solutions