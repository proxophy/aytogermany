from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Pair:
    l: str
    r: str

    def __repr__(self) -> str:
        return f"Pair('{self.l}', '{self.r}')"

    def __iter__(self):
        return iter((self.l, self.r))


class Solution:
    pairs: set[Pair] = set()

    def __init__(self, pairs: set[Pair] = set()):
        self.pairs = pairs

    def __repr__(self) -> str:
        return "(" + ", ".join(map(str, self.pairs)) + ")"

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)

    def __eq__(self, other):
        if isinstance(other, Solution):
            return self.pairs == other.pairs
        return False

    # def get_candidates(self) -> tuple[tuple[str, ...], tuple[tuple[str, ...]]]:
    #     ls, rs = zip(*self.pairs)
    #     return ls, rs

    def intersectionlength(self, other) -> int:
        if isinstance(other, Solution):
            return len(set(self.pairs) & set(other.pairs))
        elif isinstance(other, Sequence):
            return len(set(self.pairs) & set(other))
        else:
            raise NotImplementedError

    def intersection(self, other: "Solution"):
        return Solution(self.pairs & other.pairs)

    def union(self, other: "Solution"):
        return Solution(self.pairs | other.pairs)

    def issubset(self, other: "Solution"):  # type: ignore
        return set(self.pairs).issubset(set(other.pairs))

    def difference(self, other: "Solution"):
        return set(self.pairs).difference(set(other.pairs))

    def addpair(self, pair: Pair):
        return Solution(self.pairs | {pair})

    def remove(self, pair: Pair):
        self.pairs.remove(pair)


@dataclass(frozen=True)
class Night:
    pairs: tuple[Pair, ...]
    lights: int


# Matchboxes = dict[Pair, bool]
@dataclass(frozen=True)
class Matchboxes:
    episodes: Sequence[int] = ()
    pairs: Sequence[Pair] = ()
    results: Sequence[bool] = ()

    def __getitem__(self, key: Pair) -> bool:
        if key not in self.pairs:
            raise KeyError
        idx = self.pairs.index(key)
        return self.results[idx]

    def __iter__(self):
        return zip(self.episodes, self.pairs, self.results)

    def get_perfect_matches(self, end: int = -1) -> list[Pair]:
        if end == -1:
            end = max(self.episodes)
        return [
            self.pairs[i]
            for i in range(len(self.results))
            if self.results[i] and self.episodes[i] <= end
        ]

    def get_matchboxes_until_episode(self, end: int):
        c = sum(1 for e in self.episodes if e <= end)
        return Matchboxes(self.episodes[:c], self.pairs[:c], self.results[:c])

    def __contains__(self, key: Pair):
        return key in self.pairs


@dataclass(frozen=True)
class GameState:
    lefts: tuple[str]
    rights: tuple[str]
    nights: tuple[Night]
    matchboxes: Matchboxes


if __name__ == "__main__":
    p1 = Pair("Laurenz", "Joena")
    p2 = Pair("Raul", "Michelle")
    sol = Solution({p1, p2})
    myset = {p1, Pair("Raul", "Emma")}
    myset2 = {p1, p2}
    print(myset | myset2)
    mb = Matchboxes([0, 1], [p1, p2], [True, False])
    for s in sol:
        print(s)
