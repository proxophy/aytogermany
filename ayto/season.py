import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

from ayto.codec import Codec, Mask, RightBits

Person = int
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

    def get_perfect_matches(
        self, episode: int = 9, include_night: bool = False
    ) -> list[Pair]:
        pm_pairs: list[Pair] = []
        for w, p, r, bf in self:
            if w > episode:
                break
            if not include_night and w == episode and not bf:
                break
            if r:
                pm_pairs.append(p)
        return pm_pairs

    def get_no_matches(
        self, episode: int = 9, include_night: bool = False
    ) -> list[Pair]:
        no_match_pairs: list[Pair] = []
        for w, p, r, bf in self:
            if w > episode:
                break
            if not include_night and w == episode and not bf:
                break
            if not r:
                no_match_pairs.append(p)
        return no_match_pairs

    def get_matchboxes_until_week(self, episode: int, include_night: bool):
        weeks: list[int] = []
        pairs: list[Pair] = []
        results: list[bool] = []
        bfs: list[bool] = []
        for w, p, r, bf in self:
            if w > episode:
                break
            if not include_night and w == episode and not bf:
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
    lefts_names: tuple[str, ...]
    rights_names: tuple[str, ...]
    nights: tuple[Night, ...]
    matchboxes: Matchboxes
    solution: Solution = frozenset()
    match_model: str = "base"
    mm_person: str | None = None

    mm_known_from_week: int = 0  # from which episode mm_person is known
    matchbox_reveals_dm: bool = (
        True  # if one part of double match is revealed, the other one is revealed immediately after
    )
    double_match_pair_names: tuple[str, str] | None = None
    double_match_pair_known: int = 10  # from which episode double match pair is known
    extra_left_name: str | None = None  # left that is part of double match for right

    def __post_init__(self):
        # todo: write / put together validation for season data
        if not (0 <= len(self.nights) <= 10 and 0 <= len(self.matchboxes)):
            raise ValueError(
                f"Invalid number of nights or matchboxes {len(self.nights)} {len(self.matchboxes)}"
            )
        if len(set(self.lefts_names)) != len(self.lefts_names) or len(
            set(self.rights_names)
        ) != len(self.rights_names):
            raise ValueError("Duplicate names in lefts or rights")
        if self.double_match_pair_names and self.match_model != "double_match_pair":
            raise ValueError(
                f"double_match_pair is set but match_model is {self.match_model} (it should be 'double_match_pair')"
            )
        if self.extra_left_name and self.match_model != "double_match_for_right":
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
        return min(len(self.lefts_names), len(self.rights_names)) - (
            1 if self.match_model == "double_match_for_right" else 0
        )

    @property
    def double_match_pair(self) -> tuple[Person, Person] | None:
        if not self.double_match_pair_names:
            return None
        return (
            self.encode_right(self.double_match_pair_names[0]),
            self.encode_right(self.double_match_pair_names[1]),
        )

    @property
    def lefts(self):
        return list(range(len(self.lefts_names)))

    @property
    def rights(self):
        return list(range(len(self.rights_names)))

    @property
    def extra_left(self) -> Person | None:
        if self.extra_left_name:
            return self.encode_left(self.extra_left_name)
        return None

    def get_mm(self, episode: int = 9) -> Person | None:
        episode = max(0, min(episode, self.num_weeks - 1))
        if self.mm_person:
            return (
                self.encode_right(self.mm_person)
                if episode >= self.mm_known_from_week
                else None
            )
        else:
            return None

    def get_nights(
        self, episode: int = 9, include_night: bool = True
    ) -> Sequence[Night]:
        episode = max(0, episode) + (1 if include_night else 0)
        return self.nights[:episode]

    def get_matchboxes(
        self, episode: int = 9, include_night: bool = True
    ) -> Matchboxes:
        return self.matchboxes.get_matchboxes_until_week(episode, include_night)

    def get_pms(self, episode: int = 9, include_night: bool = True) -> list[Pair]:
        return self.matchboxes.get_perfect_matches(episode, include_night)

    def get_black_out_nights(self, episode: int = 9, include_night: bool = True):
        stop = max(0, episode) + (1 if include_night else 0)

        bonights: list[int] = []
        num_known_pms = 0
        pmls: list[Person] = []
        pms: set[Pair] = set()
        known_pms_per_ep = [0 for _ in range(episode + 1)]
        for ep, (l, r), res, bf in self.matchboxes:
            if ep > episode:
                break
            if ep == episode and not include_night and not bf:
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
        if l not in self.lefts_names:
            raise ValueError(f"Cannot encode {l} since it's not in lefts")
        return self.lefts_names.index(l)

    def decode_left(self, l: Person) -> str:
        return self.lefts_names[l]

    def encode_right(self, r: str) -> Person:
        if r not in self.rights_names:
            raise ValueError(f"Cannot encode {r} since it's not in rights")
        return self.rights_names.index(r)

    def decode_right(self, r: Person) -> str:
        return self.rights_names[r]

    def encode_pair(self, l: str, r: str) -> Pair:
        return (self.encode_left(l), self.encode_right(r))

    def decode_pair(self, p: Pair) -> tuple[str, str]:
        return (self.decode_left(p[0]), self.decode_right(p[1]))

    def decode_solution(self, sol: SolutionType):
        return {self.decode_pair(p) for p in sol}


def random_night_for_sol(sol: Solution, lefts: list[Person], rights: list[Person]):
    perm = random.sample(rights, len(lefts))
    pairs = {(l, r) for l, r in zip(lefts, perm)}
    lights = len(sol & pairs)
    return Night((pairs), lights)


class Knowledge:

    def __init__(self, season: Season, episode: int, include_night: bool):
        self.episode: int = episode
        self.include_night: bool = include_night

        self.nights: Sequence[Night] = season.get_nights(episode, include_night)
        self.matchboxes: Matchboxes = season.get_matchboxes(episode, include_night)
        self.perfect_matches: list[Pair] = self.matchboxes.get_perfect_matches()
        self.mb_no_matches: list[Pair] = self.matchboxes.get_no_matches()
        self.black_out_nights: list[int] = season.get_black_out_nights(
            episode, include_night
        )
        self.bp_no_matches: list[Pair] = []
        for bo in self.black_out_nights:
            for p in self.nights[bo].pairs:
                if p in self.perfect_matches:
                    continue
                self.bp_no_matches.append(p)

        self.mm: Person | None = season.get_mm(episode)

    def get_sitting_no_matches(self, sol: Solution) -> set[Pair]:
        nights = self.nights
        sitting_nomatches: set[Pair] = set()

        for night in nights:
            pl: int = len(sol & night.pairs)
            if pl < night.lights:
                continue
            for p in night.pairs - sol:
                sitting_nomatches.add(p)
        return sitting_nomatches


class BitMaskKnowledge(Knowledge):
    def __init__(self, season: Season, episode: int, include_night: bool):
        super().__init__(season, episode, include_night)
        self.codec: Codec = Codec(len(season.lefts), len(season.rights))
        self.mb_no_matches_mask: Mask = self.codec.encode(self.mb_no_matches)
        self.bp_no_matches_mask: Mask = self.codec.encode(self.bp_no_matches)
        self.perfect_matches_mask: Mask = self.codec.encode(self.perfect_matches)
        self.mm_mask: RightBits = 0 if not self.mm else 1 << self.mm
        self.night_masks: list[tuple[Mask, int]] = [
            (self.codec.encode(night.pairs), night.lights) for night in self.nights
        ]

    def get_sitting_no_matches_mask(self, sol: int) -> int:
        mask = 0
        for pairs, lights in self.night_masks:
            pl: int = (sol & pairs).bit_count()
            if pl < lights:
                continue
            mask |= pairs & ~sol
        return mask
