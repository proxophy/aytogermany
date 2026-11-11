"""Bitmask representation of a solution: bit (left_index * R + right_index) is set <=> the pair is a match."""

from collections.abc import Iterable

Mask = int
LeftBits = int
RightBits = int
Person = int
Pair = tuple[Person, Person]


def iter_bits(x: int):
    """Yield every set bit of x as its own mask (lowest first)."""
    while x:
        low = x & -x
        yield low
        x ^= low


class Codec:
    def __init__(self, n_lefts: int, n_rights: int):
        self.L: int = n_lefts
        self.R: int = n_rights
        self.full_row: RightBits = (1 << self.R) - 1  # R ones
        self.row: list[Mask] = [
            self.full_row << (i * self.R) for i in range(self.L)
        ]  # row[l]: all pairs (l, *)
        self.col: list[Mask] = [
            sum(1 << (i * self.R + j) for i in range(self.L))  # all pairs of right j
            for j in range(self.R)
        ]  # col[r]: all pairs (*, r)
        self.pairs: list[Pair] = [
            (l, r) for l in range(n_lefts) for r in range(n_rights)
        ]
        self.first_col: Mask = sum(1 << (l * self.R) for l in range(self.L))
        self.full_mask: Mask = (1 << self.R * self.L) - 1

    # ---- conversion -------------------------------------------------------------------
    def bit(self, l: int, r: int) -> Mask:
        if l >= self.L or r >= self.R:
            raise ValueError(
                f"Cannot encode {(l, r)} for a codec with L = {self.L}, R = {self.R}"
            )
        return 1 << (l * self.R + r)

    def encode(self, pairs: Iterable[Pair]) -> Mask:
        m = 0
        for l, r in pairs:
            m |= self.bit(l, r)
        return m

    def decode(self, m: Mask) -> frozenset[Pair]:
        return frozenset(self.pairs[b.bit_length() - 1] for b in iter_bits(m))

    def pair_of_bit(self, b: Mask) -> Pair:
        return self.pairs[b.bit_length() - 1]

    # ---- "which candidates are already in the solution?" -------------------------------
    def rights_in(self, m: Mask) -> RightBits:
        """Bitset over rights (R bits): every right that has at least one partner."""
        out = 0
        for i in range(self.L):
            out |= (m >> (i * self.R)) & self.full_row
        return out

    def lefts_in(self, m: Mask) -> LeftBits:
        """Bitset over lefts (L bits): every left that has at least one partner."""
        out = 0
        for i, row in enumerate(self.row):
            if m & row:
                out |= 1 << i
        return out

    def l_in_left_bits(self, lg: LeftBits, l: int) -> LeftBits:
        return (1 << l) & lg

    def r_in_right_bits(self, rg: RightBits, r: int) -> RightBits:
        return (1 << r) & rg

    def free_rights(self, m: Mask) -> RightBits:
        """Bitset over rights: the rights that still have NO partner."""
        return ~self.rights_in(m) & self.full_row

    def multi_lefts(self, m: Mask) -> LeftBits:
        """Bitset over lefts: every left with MORE than one partner (row & (row-1) != 0 <=> at least 2 bits)."""
        out = 0
        for i in range(self.L):
            row = (m >> (i * self.R)) & self.full_row
            if row & (row - 1):
                out |= 1 << i
        return out

    def multi_rights(self, m: Mask) -> RightBits:
        once = twice = 0
        full = self.full_row
        for _ in range(self.L):
            row = m & full
            twice |= once & row  # already seen in an earlier row, and again here
            once |= row
            m >>= self.R
        return twice

    def multi_for_right(self, m: Mask) -> bool:
        for mask in self.col:
            partners = m & mask
            if partners & (partners - 1):
                return True
        return False

    def max_mm_size(self, m: Mask) -> int:
        """Largest number of partners of a single left."""
        return max(
            ((m >> (i * self.R)) & self.full_row).bit_count() for i in range(self.L)
        )

    # ---- "who is the partner of X?" ----------------------------------------------------
    def partners_of_left(self, m: Mask, l: int) -> RightBits:
        """Bitset over rights (R bits): the partner(s) of left l."""
        return (m >> (l * self.R)) & self.full_row

    def partners_of_right(self, m: Mask, r: int) -> Mask:
        """Pair-bits of right r that are set: each one is a (left, r) match."""
        return m & self.col[r]

    def left_of_pair_bit(self, b: Mask) -> Person:
        return (b.bit_length() - 1) // self.R

    def right_of_pair_bit(self, b: Mask) -> Person:
        return (b.bit_length() - 1) % self.R

    # ---- names <-> bit sets (only needed at the edges) ----------------------------------
    @staticmethod
    def members(bits: LeftBits | RightBits) -> list[Person]:
        """Indices in a set of lefts or in a set of rights."""
        return [b.bit_length() - 1 for b in iter_bits(bits)]
