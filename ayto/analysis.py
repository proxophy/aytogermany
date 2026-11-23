from collections import Counter

import pandas as pd

from ayto.base import SolutionType, Solver
from ayto.season import Season, mm_left


class SolutionSpace[S: SolutionType, T]:
    """Solution space based on a solver and a list of solutions"""

    sols: list[S]
    numsols: int
    lefts: list[str]
    rights: list[str]

    def __init__(self, solver: Solver[S, T], sols: list[S]):
        self.sols = sols
        self.numsols = len(sols)
        if len(sols) == 0:
            raise ValueError("No solutions provided")
        self.season: Season = solver.season
        self.solver: Solver[S, T] = solver
        self.lefts = sorted(self.season.lefts_names)
        self.rights = sorted(self.season.rights_names)
        self.all_pairs: list[tuple[str, str]] = [
            (l, r) for l in self.lefts for r in self.rights
        ]
        self.pairs_counter: Counter[tuple[str, str]] = Counter(
            self.season.decode_pair(p) for s in self.sols for p in solver.to_pairs(s)
        )

        for p in self.all_pairs:
            if p not in self.pairs_counter:
                self.pairs_counter[p] = 0

    def get_pair_probs_dict(self):
        return {
            (l, r): self.pairs_counter[(l, r)] / self.numsols * 100
            for (l, r) in self.pairs_counter
        }

    def get_most_likely_pair(self) -> tuple[str, str]:
        probs_dict = self.get_pair_probs_dict()
        return max(probs_dict, key=lambda x: probs_dict.get(x, 0))

    def get_no_matches(self) -> list[tuple[str, str]]:
        return [p for p in self.pairs_counter if self.pairs_counter[p] == 0]

    def get_perfect_matches(self) -> list[tuple[str, str]]:
        return [p for p in self.pairs_counter if self.pairs_counter[p] == self.numsols]

    def get_no_matches_for_l(self, l: str) -> list[str]:
        if l not in self.lefts:
            raise ValueError(f"{l} is not a candidate in these solutions")
        return [r for r in self.rights if self.pairs_counter[(l, r)] == 0]

    def get_no_matches_for_r(self, r: str) -> list[str]:
        if r not in self.rights:
            raise ValueError(f"{r} is not a candidate in these solutions")
        return [l for l in self.lefts if self.pairs_counter[(l, r)] == 0]

    def get_mm_probs(self):
        mm_person = [
            self.season.decode_left(
                mm_left(self.solver.to_pairs(s))  #  pyright: ignore[reportArgumentType]
            ) 
            for s in self.sols
        ]
        return [
            (l, round(v / len(self.sols) * 100, 1))
            for (l, v) in Counter(mm_person).items()
        ]


def analyze_solutions[S: SolutionType, T](sol_space: SolutionSpace[S, T]) -> str:
    """Returns short analysis of solutions in a solution space in German"""

    rs = (
        f"Anzahl Möglichkeiten: {sol_space.numsols}\n"
        + "Perfect Matches\n"
        + str(sol_space.get_perfect_matches())
        + "\n"
        + "No-Matches:"
        + "\n".join(
            l + ": " + str(sol_space.get_no_matches_for_l(l)) for l in sol_space.lefts
        )
        + "\n"
        + f"Person mit Mehrfachmatch: {sol_space.get_mm_probs()}"
    )

    return rs


def matching_night_probs[S: SolutionType, T](solver: Solver[S, T], episode: int):
    """Returns the probability distribution for the number of lights in an episode"""
    sols_before_night = solver.solve(episode, False)
    night = solver.season.nights[episode].pairs
    if len(sols_before_night) == 0:
        raise ValueError(f"no solutions left before the night of episode {episode}")
    poslights = Counter(
        [len(night & solver.to_pairs(sol)) for sol in sols_before_night]
    )
    return [
        (i, round(poslights.get(i, 0) / len(sols_before_night) * 100, 2))
        for i in range(solver.season.max_lights + 1)
    ]


def candidate_probs[S: SolutionType, T](solver: Solver[S, T], sols: list[S], sol: S):
    """The probability distribution of the lights if sol is the season's solution"""
    # todo: consider multiple matches
    lights = Counter([solver.size(s & sol) for s in sols])
    probs = [lights.get(i, 0) / len(sols) for i in range(solver.size(sol) + 1)]

    return probs


def df_for_probs[S: SolutionType, T](sol_space: SolutionSpace[S, T]):
    """Create a pandas dataframe for the pair probabilities """
    probs = sol_space.get_pair_probs_dict()
    df = pd.Series(probs).unstack(fill_value=0)
    return df


def plot_probs[S: SolutionType, T](sol_space: SolutionSpace[S, T]):
    """Create a seaborn plot for the pair probabilities """
    import matplotlib.pyplot as plt
    import seaborn as sns

    probs = sol_space.get_pair_probs_dict()
    df = pd.Series(probs).unstack(fill_value=0)
    my_cmap = sns.light_palette((0.2, 0.7, 0.2), as_cmap=True)
    my_cmap.set_under((1, 0.7, 0.7))
    my_cmap.set_over((0.2, 0.8, 0.2))
    _ = sns.heatmap(  # pyright: ignore[reportUnknownMemberType]
        df, vmin=1e-5, vmax=100 - 1e-5, cmap=my_cmap, annot=True, fmt=".0f"
    )
    plt.tight_layout()
    plt.show()  # pyright: ignore[reportUnknownMemberType]
