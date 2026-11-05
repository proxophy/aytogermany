import time
from collections import Counter
from collections.abc import Callable
from functools import wraps

import pandas as pd

from .season import Pair, Solution, mm_left
from .solver import Solver


class SolutionSpace:
    sols: list[Solution]
    numsols: int
    lefts: list[str]
    rights: list[str]

    def __init__(self, sols: list[Solution]):
        self.sols = sols
        self.numsols = len(sols)
        if len(sols) == 0:
            raise ValueError("No solutions provided")
        clefts, crights = zip(*sols[0])
        self.lefts = sorted(set(clefts))
        self.rights = sorted(set(crights))
        self.allpairs: list[Pair] = [(l, r) for l in self.lefts for r in self.rights]
        self.pairs_counter: Counter[Pair] = Counter([p for s in self.sols for p in s])
        for p in self.allpairs:
            if p not in self.pairs_counter:
                self.pairs_counter[p] = 0

    def get_pair_probs_dict(self):
        return {
            (l, r): round(self.pairs_counter[(l, r)] / self.numsols * 100, 1)
            for (l, r) in self.pairs_counter
        }

    def get_most_likely_pair(self) -> Pair:
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
        mm = [mm_left(s) for s in self.sols]
        return [
            (l, round(v / len(self.sols) * 100, 1)) for (l, v) in Counter(mm).items()
        ]


def index_to_episode(i: int) -> int:
    return 2 * (i + 1) + 1


def analyze_solutions(sols: list[Solution]):
    solspace = SolutionSpace(sols)

    print(f"Anzahl Möglichkeiten: {solspace.numsols}")
    print("Perfect Matches")
    print(solspace.get_perfect_matches())
    print("No-Matches:")
    for l in solspace.lefts:
        print(f"{l}: {solspace.get_no_matches_for_l(l)}")

    print("Person mit Doppelmatch:", solspace.get_mm_probs())

    return solspace.get_pair_probs_dict()


def matching_night_probs(solver: Solver, episode: int):
    beforenight = solver.solve(episode, False)
    night = solver.season.get_nights(episode)[-1].pairs
    if len(beforenight) == 0:
        raise ValueError(f"no solutions left before the night of episode {episode}")
    poslights = Counter([len(set(night).intersection(sol)) for sol in beforenight])
    return [
        (i, round(poslights.get(i, 0) / len(beforenight) * 100, 2))
        for i in range(solver.season.max_lights + 1)
    ]


def sol_probs(sols: list[Solution], sol: Solution):
    lights = Counter([len(s.intersection(sol)) for s in sols])
    probs = [round(lights.get(i, 0) / len(sols) * 100, 2) for i in range(len(sol) + 1)]
    import statistics

    return (
        probs,
        statistics.median(filter(lambda x: x > 0, probs)),
        max(filter(lambda x: x > 0, probs)),
    )


def df_for_probs(sols: list[Solution]):
    solspace = SolutionSpace(sols)
    probs = solspace.get_pair_probs_dict()
    df = pd.Series(probs).unstack(fill_value=0)
    return df


def plot_probs(sols: list[Solution]):
    import matplotlib.pyplot as plt
    import seaborn as sns

    solspace = SolutionSpace(sols)
    probs = solspace.get_pair_probs_dict()
    df = pd.Series(probs).unstack(fill_value=0)
    my_cmap = sns.light_palette((0.2, 0.7, 0.2), as_cmap=True)
    my_cmap.set_under((1, 0.7, 0.7))
    my_cmap.set_over((0.2, 0.8, 0.2))
    _ = sns.heatmap(  # pyright: ignore[reportUnknownMemberType]
        df, vmin=1e-5, vmax=100 - 1e-5, cmap=my_cmap, annot=True, fmt=".0f"
    )
    plt.tight_layout()
    plt.show()  # pyright: ignore[reportUnknownMemberType]


def time_it[**P, R](inner: Callable[P, R]) -> Callable[P, R]:
    @wraps(inner)
    def c_inner(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.time()
        res = inner(*args, **kwargs)
        end = time.time()
        print(f"=== time needed for {inner.__name__}: {(end - start):0.3f}s ===")
        return res

    return c_inner
