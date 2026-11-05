import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

Person = str
# class Person(str):
#     pass
Pair = tuple[Person, Person]
Solution = frozenset[Pair]
SolutionType = Solution | set[Pair]

# class Solution(frozenset[Pair]):



def has_dm_for_right(sol: SolutionType) -> bool:
    """Return whether solution has a double match for a right"""
    rs: list[Person] = []
    for _, r in sol:
        if r in rs:
            return True
        rs.append(r)
    return False


def get_candidates(sol: SolutionType) -> tuple[set[Person], set[Person], int]:
    """Return sets of lefts and rights in solution and size of the maximum multuple match"""
    if len(sol) > 0:
        g_lefts, g_rights = zip(*sol)
        g_lefts, g_rights = list(g_lefts), list(g_rights)
    else:
        return set(), set(), 0
    d = dict_rep(sol)
    return set(g_lefts), set(g_rights), max(map(len, d.values()))


def mm_left(sol: SolutionType) -> Person | None:
    """Return first left with multiple match or `None` if there isn't any"""
    ls: set[Person] = set()
    for l, _ in sol:
        if l in ls:
            return l
        ls.add(l)
    return None


def has_two_dms(sol: SolutionType) -> bool:
    """Return wheter there are two double matches in a solution"""
    sol_dict = dict_rep(sol)
    # lefts with multiple partners
    mult_lefts = [l for l, rs in sol_dict.items() if len(rs) > 1]
    return len(mult_lefts) == 2


def dict_rep(sol: SolutionType) -> dict[Person, list[Person]]:
    """Return dict representation of set of pairs"""
    d: dict[Person, list[Person]] = {}
    for l, r in sol:
        if l in d:
            d[l].append(r)
        else:
            d[l] = [r]
    return d


@dataclass(frozen=True)
class Night:
    pairs: frozenset[Pair] | set[Pair]
    lights: int

    def __post_init__(self):
        object.__setattr__(self, "pairs", frozenset(self.pairs))
        if not 0 <= self.lights <= len(self.pairs):
            raise ValueError("Invalid number of lights")


# Matchboxes = dict[Pair, bool]
@dataclass(frozen=True)
class Matchboxes:
    weeks: tuple[int, ...] = ()
    pairs: tuple[Pair, ...] = ()
    results: tuple[bool, ...] = ()
    before_night: tuple[bool, ...] = ()

    def __post_init__(self):
        if not (
            len(self.weeks)
            == len(self.pairs)
            == len(self.results)
            == len(self.before_night)
        ):
            raise ValueError(
                len(self.weeks),
                len(self.pairs),
                len(self.results),
                len(self.before_night),
            )
        l = self.weeks
        if not all(l[i] <= l[i + 1] for i in range(len(l) - 1)):
            raise ValueError("weeks in matchbox list must be increasing")

    def __getitem__(self, key: Pair) -> bool:
        if key not in self.pairs:
            raise KeyError
        idx = self.pairs.index(key)
        return self.results[idx]

    def __iter__(self):
        return zip(self.weeks, self.pairs, self.results, self.before_night)

    def __len__(self):
        return len(self.weeks)

    def __contains__(self, key: Pair):
        return key in self.pairs

    @override
    def __str__(self):
        s = ""
        for w, p, r, bf in self:
            s += f"{w}: {p} -> {r}, bf={bf}\n"
        return s

    def get_perfect_matches(self, end: int = 9, include_night: bool = False):
        pm_pairs: list[Pair] = []
        for w, p, r, bf in self:
            if w > end:
                break
            if not include_night and w == end and not bf:
                break
            if r:
                pm_pairs.append(p)
        return pm_pairs

    def get_matchboxes_until_week(self, end: int, include_night: bool):
        weeks: list[int] = []
        pairs: list[Pair] = []
        results: list[bool] = []
        bfs: list[bool] = []
        for w, p, r, bf in self:
            if w > end:
                break
            if not include_night and w == end and not bf:
                break
            weeks.append(w)
            pairs.append(p)
            results.append(r)
            bfs.append(bf)

        return Matchboxes(tuple(weeks), tuple(pairs), tuple(results), tuple(bfs))


def matchboxes_from_list(data: list[tuple[int, Pair, bool, bool]]):
    episodes, pairs, results, bfs = zip(*data)
    return Matchboxes(episodes, pairs, results, bfs)


@dataclass(frozen=True)
class Season:
    name: str
    lefts: tuple[str, ...]
    rights: tuple[str, ...]
    nights: tuple[Night, ...]
    matchboxes: Matchboxes
    solution: Solution = Solution()  # noqa: RUF009
    match_model: str = "base"
    mm_person: Person | None = None

    mm_known_from_week: int = 0  # from which episode mm_person is known
    matchbox_reveals_dm: bool = (
        True  # if one part of double match is revealed, the other one is revealed immediately after
    )
    double_match_pair: tuple[Person, Person] | None = None
    double_match_pair_known: int = 10  # from which episode double match pair is known
    extra_left: Person | None = None  # left that is part of double match for right

    def __post_init__(self):
        # todo: write / put together validation for season data
        if not (0 <= len(self.nights) <= 10 and 0 <= len(self.matchboxes)):
            raise ValueError(
                f"Invalid number of nights or matchboxes {len(self.nights)} {len(self.matchboxes)}"
            )
        if len(set(self.lefts)) != len(self.lefts) or len(set(self.rights)) != len(
            self.rights
        ):
            raise ValueError("Duplicate names in lefts or rights")
        if self.double_match_pair and self.match_model != "double_match_pair":
            raise ValueError(
                f"double_match_pair is set but match_model is {self.match_model} (it should be 'double_match_pair')"
            )
        if self.extra_left and self.match_model != "double_match_for_right":
            raise ValueError(
                f"extra_left is set but match_model is {self.match_model} (it should be 'double_match_for_right')"
            )

    @property
    def num_weeks(self) -> int:
        return len(self.nights)

    @property
    def two_dms(self) -> bool:
        return self.match_model == "two_double_matches"

    @property
    def double_match_for_right(self) -> bool:
        return self.match_model == "double_match_for_right"

    @property
    def max_mm_size(self) -> int:
        return 3 if self.match_model == "triple_match" else 2

    @property
    def num_matches(self) -> int:
        if self.match_model in (
            "triple_match",
            "two_double_matches",
            "double_match_for_right",
        ):
            return self.max_lights + 2
        else:
            return self.max_lights + 1

    @property
    def max_lights(self) -> int:
        return min(len(self.lefts), len(self.rights)) - (
            1 if self.match_model == "double_match_for_right" else 0
        )

    def get_mm(self, end: int = 9) -> Person | None:
        end = max(0, min(end, self.num_weeks - 1))
        return self.mm_person if end >= self.mm_known_from_week else None

    def get_nights(self, end: int = 9, include_night: bool = True) -> Sequence[Night]:
        end = max(0, end) + (1 if include_night else 0)
        return self.nights[:end]

    def get_matchboxes(self, end: int = 9, include_night: bool = True) -> Matchboxes:
        return self.matchboxes.get_matchboxes_until_week(end, include_night)

    def get_pms(self, end: int = 9, include_night: bool = True) -> list[Pair]:
        return self.matchboxes.get_perfect_matches(end, include_night)

    def get_sitting_no_matches(
        self, sol: Solution, end: int = 9, include_night: bool = True
    ) -> set[Pair]:
        nights = self.get_nights(end, include_night)
        sitting_nomatches: set[Pair] = set()

        for night in nights:
            pl: int = len(sol & night.pairs)
            if pl < night.lights:
                continue
            for p in night.pairs - sol:
                sitting_nomatches.add(p)
        return sitting_nomatches

    def get_black_out_nights(self, end: int = 9, include_night: bool = True):
        stop = max(0, end) + (1 if include_night else 0)
        

        bonights: list[int] = []
        num_known_pms = 0
        pmls: list[Person] = []
        pms: set[Pair] = set()
        known_pms_per_ep = [0 for _ in range(end + 1)]
        for ep, (l, r), res, bf in self.matchboxes:
            if ep > end:
                break
            if ep == end and not include_night and not bf:
                break  # skip matchbox after last night
            if res and l not in pmls:
                # add one perfect match per person, ignoring double matches
                num_known_pms += 1
                pmls.append(l)
                pms.add((l, r))
            known_pms_per_ep[ep] = num_known_pms


        for ep in range(min(self.num_weeks, stop)):
            known_pms_per_ep[ep] = max(
                known_pms_per_ep[ep], known_pms_per_ep[ep - 1] if ep > 0 else 0
            )
            if known_pms_per_ep[ep] == self.nights[ep].lights:
                bonights.append(ep)
        return bonights

    def encode_left(self, l: str) -> Person:
        return l  # pyright: ignore[reportReturnType]


def random_night_for_sol(sol: Solution, lefts: list[Person], rights: list[Person]):
    perm = random.sample(rights, len(lefts))
    pairs = {(l, r) for l, r in zip(lefts, perm)}
    lights = len(sol & pairs)
    return Night((pairs), lights)
