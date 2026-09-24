from dataclasses import dataclass
from typing import Sequence, Generator
from collections import Counter

Pair = tuple[str, str]
Solution = frozenset[Pair]


def double_match_for_right(sol: Solution | set[Pair]):
    rs = []
    for _, r in sol:
        if r in rs:
            return True
        rs.append(r)
    return False


def get_candidates(sol: Solution | set[Pair]):
    if len(sol) > 0:
        g_lefts, g_rights = zip(*sol)
        g_lefts, g_rights = list(g_lefts), list(g_rights)
    else:
        return set(), set(), 0
    d = dict_rep(sol)
    return set(g_lefts), set(g_rights), max(map(len,d.values()))


def mm_left(sol: Solution | set[Pair]) -> str | None:
    ls = set()
    for l, _ in sol:
        if l in ls:
            return l
        ls.add(l)
    return None


def has_two_dms(sol: Solution | set[Pair]) -> bool:
    sol_dict = dict_rep(sol)
    # lefts with multiple partners
    mult_lefts = [l for l, rs in sol_dict.items() if len(rs) > 1]
    return len(mult_lefts) == 2

def dict_rep(sol: Solution | set[Pair]) -> dict[str, list[str]]:
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
    num_matches: int = 11
    double_match_pair: tuple[str, str] | None = None
    double_match_pair_known: int = 7  # after which episode mm is known
    mm_known_after_week: int = 0 # after which episode mm is known
    max_mm_size: int = 2 # max size of multiple match pair
    mb_reveals_dm: bool = True # if one part of double match is revealed, the other one is revealed immediately after


    def __post_init__(self):
        if not (0 <= len(self.nights) <= 10 and 0 <= len(self.matchboxes)):
            raise ValueError(
                f"Invalid number of nights or matchboxes {len(self.nights)} {len(self.matchboxes)}"
            )
        if len(set(self.lefts)) != len(self.lefts) or len(set(self.rights)) != len(self.rights):
            raise ValueError(f"Duplicate names in lefts or rights")

    @property
    def num_weeks(self) -> int:
        return len(self.nights)

    def get_mm(self, end: int = 10) -> str|None:
        end = max(0, min(end, 10))
        return self.mm if end >= self.mm_known_after_week else None

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

    def get_black_out_nights(self):
        bonights = set()
        num_known_pms = 0
        pmls = []
        for ep, (l, r), res in self.matchboxes:
            if res and l not in pmls:
                num_known_pms += 1
                pmls.append(l)
            if num_known_pms == self.nights[ep].lights:
                bonights.add(ep)
        return bonights