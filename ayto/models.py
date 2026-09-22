from dataclasses import dataclass
from typing import Sequence, Generator

Pair = tuple[str, str]
Solution = frozenset[Pair]


def double_match_for_right(sol: Solution):
    rs = []
    for _, r in sol:
        if r in rs:
            return True
        rs.append(r)
    return False


def get_candidates(sol: Solution):
    if len(sol) > 0:
        g_lefts, g_rights = zip(*sol)
        g_lefts, g_rights = list(g_lefts), list(g_rights)
    else:
        g_lefts, g_rights = set(), set()
    return set(g_lefts), set(g_rights), len(g_lefts) - len(set(g_lefts))


def mm_left(sol: Solution) -> str | None:
    ls = set()
    for l, _ in sol:
        if l in ls:
            return l
        ls.add(l)
    return None


def dict_rep(sol: Solution) -> dict[str, list[str]]:
    d = {}
    for l, r in sol:
        if l in d:
            d[l].append(r)
        else:
            d[l] = [r]
    return d


@dataclass(frozen=True)
class Night:
    pairs: set[Pair]
    lights: int

    def __post_init__(self):
        if not 0 <= self.lights <= len(self.pairs):
            raise ValueError("Invalid number of lights")


# Matchboxes = dict[Pair, bool]
@dataclass(frozen=True)
class Matchboxes:
    weeks: tuple[int, ...] = ()
    pairs: tuple[Pair, ...] = ()
    results: tuple[bool, ...] = ()

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
    lefts: tuple[str, ...]
    rights: tuple[str, ...]
    nights: tuple[Night, ...]
    matchboxes: Matchboxes
    solution: Solution | None = None
    mm: str | None = None
    double_match_pair: tuple[str, str] | None = None
    double_match_pair_known: int = 7

    def __post_init__(self):
        if not (0 <= len(self.nights) <= 10 and 0 <= len(self.matchboxes)):
            raise ValueError(
                f"Invalid number of nights or matchboxes {len(self.nights)} {len(self.matchboxes)}"
            )

    @property
    # TODO: give as parameter
    def num_matches(self) -> int:
        return max(len(self.lefts), len(self.rights)) + (
            1 if self.name == "vip2026" else 0
        )

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


    def get_sitting_no_matches(self, sol: Solution,  end: int = 10, include_night: bool = True) -> set[Pair]:
        nights = self.get_nights(end)
        if not include_night:
            nights = nights[:-1]
        sitting_nomatches = set()

        for night in nights:
            pl: int = len(sol & night.pairs)
            if pl < night.lights:
                continue
            for p in night.pairs - sol:
                sitting_nomatches.add(p)
        return sitting_nomatches
