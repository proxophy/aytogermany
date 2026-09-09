from dataclasses import dataclass


@dataclass(frozen=True)
class Pair:
    left: str
    right: str

    def __repr__(self) -> str:
        return f"({self.left}, {self.right})"


@dataclass(frozen=True)
class Solution:
    pairs: tuple[Pair, ...]

    def __repr__(self) -> str:
        return ", ".join(map(str, self.pairs))

@dataclass(frozen=True)
class Night:
    pairs: tuple[Pair, ...]
    lights: int

Matchboxes = dict[Pair, bool]

@dataclass(frozen=True)
class GameState:
    lefts: tuple[str]
    rights: tuple[str]
    nights: tuple[Night]
    matchboxes: Matchboxes


if __name__ == "__main__":
    p1 = Pair("Laurenz", "Joena")
    p2 = Pair("Raul", "Michelle")
    s = Solution((p1, p2))
    print(s)
