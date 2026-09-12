from dataclasses import dataclass
from typing import Sequence
from collections import Counter


Pair = tuple[str, str]
# @dataclass(frozen=True)
# class Pair:
#     l: str
#     r: str

#     def __repr__(self) -> str:
#         return f"Pair('{self.l}', '{self.r}')"

#     def __iter__(self):
#         return iter((self.l, self.r))

#     def tuplerep(self):
#         return (self.l, self.r)


@dataclass(frozen=True)
class Solution:
    pairs: frozenset[Pair] = frozenset()


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
        return Solution(self.pairs.difference(pair))

    def mm_left(self) -> str:
        ls, rs = zip(*self.pairs)
        return [l for (l, v) in Counter(ls).items() if v >= 2][0]


@dataclass(frozen=True)
class Night:
    pairs: tuple[Pair, ...]
    lights: int

    def __post_init__(self):
        if not 0 <= self.lights <= len(self.pairs):
            raise ValueError("Invalid number of lights")


# Matchboxes = dict[Pair, bool]
@dataclass(frozen=True)
class Matchboxes:
    episodes: Sequence[int] = ()
    pairs: Sequence[Pair] = ()
    results: Sequence[bool] = ()

    def __post_init__(self):
        if not (len(self.episodes) == len(self.pairs) == len(self.results)):
            raise ValueError
        pass

    def __getitem__(self, key: Pair) -> bool:
        if key not in self.pairs:
            raise KeyError
        idx = self.pairs.index(key)
        return self.results[idx]

    def __iter__(self):
        return zip(self.episodes, self.pairs, self.results)

    def __len__(self):
        return len(self.episodes)

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
class Season:
    name: str
    lefts: Sequence[str]
    rights: Sequence[str]
    nights: Sequence[Night]
    matchboxes: Matchboxes
    solution: Solution | None = None
    mm: str | None = None
    dmtuple: tuple[str, str] | None = None
    dmtupleknown: int = 7
    two_dms: bool = False
   

    def __post_init__(self):
        if not (0 <= len(self.nights) <= 10 and 0 <= len(self.matchboxes)):
            raise ValueError(
                f"Invalid number of nights or matchboxes {len(self.nights)} {len(self.matchboxes)}"
            )

    @property
    def nummatches(self) -> int:
        return max(len(self.lefts), len(self.rights)) + (1 if self.name == "vip2026" else 0)

    @property
    def numepisodes(self) -> int:
        return len(self.nights)

    def get_nights(self, end: int = 10) -> Sequence[Night]:
        end = max(0, min(end, 10))
        return self.nights[: (end + 1)]

    def get_matchboxes(self, end: int = 10) -> Matchboxes:
        end = max(0, min(end, 10))
        return self.matchboxes.get_matchboxes_until_episode(end)

    def get_pms(self, end: int = 10) -> list[Pair]:
        end = max(0, min(end, 10))
        return self.matchboxes.get_perfect_matches(end)


if __name__ == "__main__":
    p1 = ("Laurenz", "Joena")
    p2 = ("Raul", "Michelle")
    sol = Solution(frozenset((p1, p2)))
    myset = {p1, ("Raul", "Emma")}
    myset2 = {p1, p2}
    print(myset | myset2)
    mb = Matchboxes([0, 1], [p1, p2], [True, False])
    print(sol.mm_left())
