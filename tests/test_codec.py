"""Tests for codec.py against naive set-based reference implementations (no solver needed)."""

import random
import unittest
from collections import Counter
from random import Random
from typing import override

from ayto.codec import Codec, iter_bits

SIZES = [(10, 11), (10, 12), (11, 11)]  # (lefts, rights) of the real seasons


def random_pairs(rng : Random, L: int, R: int):
    """Random set of pairs, including lefts with several partners and the empty set."""
    k = rng.choice([0, 1, 2, 5, 8, 11, 12, 14])
    return frozenset((rng.randrange(L), rng.randrange(R)) for _ in range(k))


class TestCodec(unittest.TestCase):
    @override
    def setUp(self):
        self.rng: Random = random.Random( # pyright: ignore[reportUninitializedInstanceVariable]
            1234
        )  

    def cases(self, n=400):
        for L, R in SIZES:
            codec = Codec(L, R)
            for _ in range(n):
                yield codec, random_pairs(self.rng, L, R)

    def test_roundtrip_and_injective(self):
        for L, R in SIZES:
            codec = Codec(L, R)
            sets = {random_pairs(self.rng, L, R) for _ in range(3000)}
            masks = {codec.encode(s) for s in sets}
            self.assertEqual(len(masks), len(sets))  # different sets -> different masks
            for s in sets:
                self.assertEqual(codec.decode(codec.encode(s)), s)

    def test_candidates_in_solution(self):
        for codec, s in self.cases():
            m = codec.encode(s)
            self.assertEqual(set(Codec.members(codec.rights_in(m))), {r for _, r in s})
            self.assertEqual(set(Codec.members(codec.lefts_in(m))), {l for l, _ in s})
            self.assertEqual(
                set(Codec.members(codec.free_rights(m))),
                set(range(codec.R)) - {r for _, r in s},
            )

    def test_partners(self):
        for codec, s in self.cases():
            m = codec.encode(s)
            for l in range(codec.L):
                self.assertEqual(
                    set(Codec.members(codec.partners_of_left(m, l))),
                    {r for ll, r in s if ll == l},
                )
            for r in range(codec.R):
                lefts = {
                    codec.left_of_pair_bit(b)
                    for b in iter_bits(codec.partners_of_right(m, r))
                }
                self.assertEqual(lefts, {l for l, rr in s if rr == r})
                for b in iter_bits(codec.partners_of_right(m, r)):
                    self.assertEqual(codec.right_of_pair_bit(b), r)

    def test_multiple_matches(self):
        for codec, s in self.cases():
            m = codec.encode(s)
            counts = Counter(l for l, _ in s)
            self.assertEqual(
                set(Codec.members(codec.multi_lefts(m))),
                {l for l, c in counts.items() if c > 1},
            )
            self.assertEqual(codec.max_mm_size(m), max(counts.values(), default=0))

    def test_pair_operations(self):
        codec = Codec(10, 11)
        s = frozenset({(0, 0), (3, 5), (9, 10)})
        m = codec.encode(s)
        self.assertTrue(m & codec.bit(3, 5))
        self.assertFalse(m & codec.bit(3, 6))
        self.assertEqual(m.bit_count(), 3)  # number of pairs
        self.assertEqual(codec.pair_of_bit(codec.bit(9, 10)), (9, 10))
        night = codec.encode({(0, 0), (1, 1), (3, 5)})
        self.assertEqual((m & night).bit_count(), 2)  # "lights" of a night
        self.assertEqual(
            codec.partners_of_right(0, 4), 0
        )  # no partner -> 0, never an index of -1

    def test_bits_fit_into_two_uint64(self):
        for L, R in SIZES:
            self.assertLessEqual(L * R, 128)


if __name__ == "__main__":
    unittest.main()  # pyright: ignore[reportUnusedCallResult]
