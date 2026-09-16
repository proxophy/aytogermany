from dataclasses import dataclass
from typing import Sequence


Pair = tuple[str, str]

@dataclass(frozen=True)
class Solution:
    pairs: frozenset[Pair] = frozenset()


    def __repr__(self) -> str:
        return "(" + ", ".join(map(str, self.pairs)) + ")" # type: ignore

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)


    def intersection_length(self, night: Sequence[Pair]) -> int:
        if len(night) != 10:
            raise ValueError("intersectionlength must be called with night as arg")
        count = sum(1 for p in night if p in self.pairs)
        return count

    def intersection(self, other: "Solution"):
        return Solution(self.pairs & other.pairs)

    def union(self, other: "Solution"):
        return Solution(self.pairs | other.pairs)

    def issubset(self, other: "Solution"):  # type: ignore
        return set(self.pairs).issubset(set(other.pairs))

    def difference(self, other: "Solution"):
        return set(self.pairs).difference(set(other.pairs))

    def addpair(self, pair: Pair) -> "Solution":
        return Solution(self.pairs | {pair})

    def remove(self, pair: Pair) -> "Solution":
        return Solution(self.pairs.difference([pair]))

    def mm_left(self) -> str | None:
        ls = []
        for l,_ in self.pairs:
            if l in ls:
                return l
            ls.append(l)
        return None

    def double_match_for_right(self) -> bool:
        rs = []
        for _ , r in self.pairs:
            if r in rs:
                return True
            rs.append(r)
        return False

    def get_candidates(self):
        if len(self) > 0:
            g_lefts, g_rights = zip(*self.pairs)
            g_lefts, g_rights = list(g_lefts), list(g_rights)
        else:
            g_lefts, g_rights = set(), set()
        return set(g_lefts), set(g_rights), len(g_lefts) - len(set(g_lefts))

    def dict_rep(self) -> dict[str, list[str]]:
        d = {}
        for l, r in self.pairs:
            if l in d:
                d[l].append(r)
            else:
                d[l] = [r]
        return d



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
    weeks: tuple[int,...] = ()
    pairs: tuple[Pair,...] = ()
    results: tuple[bool,...] = ()

    def __post_init__(self):
        if not (len(self.weeks) == len(self.pairs) == len(self.results)):
            raise ValueError
        pass

    def __getitem__(self, key: Pair) -> bool:
        if key not in self.pairs:
            raise KeyError
        idx = self.pairs.index(key)
        return self.results[idx]

    def __iter__(self):
        return zip(self.weeks, self.pairs, self.results)

    def __len__(self):
        return len(self.weeks)

    def get_perfect_matches(self, end: int = -1) -> list[Pair]:
        if end == -1:
            end = max(self.weeks)
        return [
            self.pairs[i]
            for i in range(len(self.results))
            if self.results[i] and self.weeks[i] <= end
        ]

    def get_matchboxes_until_week(self, end: int):
        c = sum(1 for e in self.weeks if e <= end)
        return Matchboxes(self.weeks[:c], self.pairs[:c], self.results[:c])

    def __contains__(self, key: Pair):
        return key in self.pairs


@dataclass(frozen=True)
class Season:
    name: str
    lefts: tuple[str,...]
    rights: tuple[str,...]
    nights: tuple[Night,...]
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
    def num_matches(self) -> int:
        return max(len(self.lefts), len(self.rights)) + (1 if self.name == "vip2026" else 0)

    @property
    def num_weeks(self) -> int:
        return len(self.nights)

    @property
    def mm_known_after_week(self) -> int:
        return 5 if self.name == "normalo2026" else 0

    @property
    def max_multiple_match_size(self) -> int:
        return 3 if self.name == "normalo2024" else 2

    def get_nights(self, end: int = 10) -> Sequence[Night]:
        end = max(0, min(end, 10))
        return self.nights[: (end + 1)]

    def get_matchboxes(self, end: int = 10) -> Matchboxes:
        end = max(0, min(end, 10))
        return self.matchboxes.get_matchboxes_until_week(end)

    def get_pms(self, end: int = 10) -> list[Pair]:
        end = max(0, min(end, 10))
        return self.matchboxes.get_perfect_matches(end)


