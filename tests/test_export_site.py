"""Tests for scripts/export_site.py (chunked solving, probs table, caching).

Place in tests/ and run from the repository root, e.g.

    uv run python -m unittest tests.test_export_site -v

Uses vip2021 (about 2 million solutions in the biggest episode, ~5 s for the
whole module). get_season()/get_solver() read data/... relative to the repo root.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ayto import SolutionSpace, df_for_probs, get_season, get_solver
from scripts.export_site import (
    SCHEMA_VERSION,
    code_hash,
    episode_hash,
    export_probs,
    probs_table,
    solution_chunks,
)

SEASON = "vip2021"
CODE = code_hash()
# 0-based episodes that are cheap to solve and also solved by the old code path
EPISODES = (2, 5, 9)


def run_export(
    out: Path, *, force: bool = False, code: str = CODE, max_solve: int | None = None
) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        export_probs(SEASON, out, force, code, max_solve)


class SolutionChunksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.season = get_season(SEASON)
        cls.solver = get_solver(SEASON, True)

    def test_chunks_are_the_same_solutions_as_solve(self):
        for ep in EPISODES:
            with self.subTest(ep=ep):
                flat = [s for c in solution_chunks(self.solver, ep) for s in c]
                self.assertEqual(sorted(flat), sorted(self.solver.solve(ep, True)))

    def test_chunks_are_disjoint(self):
        # counts of the chunks may only be added up if no solution occurs twice
        for ep in EPISODES:
            with self.subTest(ep=ep):
                flat = [s for c in solution_chunks(self.solver, ep) for s in c]
                self.assertEqual(len(flat), len(set(flat)))

    def test_probs_table_without_solutions_raises(self):
        codec = self.solver.codec
        for chunks in ([], [[]]):
            with self.subTest(chunks=chunks), self.assertRaises(ValueError):
                probs_table(self.season, codec, chunks)

    def test_probs_table_is_independent_of_chunking(self):
        ep = EPISODES[0]
        sols = self.solver.solve(ep, True)
        one, n1 = probs_table(self.season, self.solver.codec, [sols])
        halves, n2 = probs_table(
            self.season,
            self.solver.codec,
            [sols[: len(sols) // 3], sols[len(sols) // 3 :]],
        )
        self.assertEqual(n1, n2)
        self.assertEqual(one, halves)


class ExportProbsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.season = get_season(SEASON)
        cls.solver = get_solver(SEASON, True)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)
        run_export(cls.out)  # the only full solve of the module
        cls.path = cls.out / "probs" / f"{SEASON}.json"
        cls.full_text = cls.path.read_text(encoding="utf-8")
        cls.full = json.loads(cls.full_text)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def fresh(self, text: str | None = None) -> Path:
        """New output directory that contains a cached probs file (the full one by default)."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        path = out / "probs" / f"{SEASON}.json"
        path.parent.mkdir(parents=True)
        path.write_text(self.full_text if text is None else text, encoding="utf-8")
        return out

    def tampered(self, *, stale: tuple[int, ...] = (), missing: tuple[int, ...] = ()) -> str:
        """Cached file where the given episodes (1-based) have a wrong hash or are missing."""
        data = json.loads(self.full_text)
        for e in stale:
            data["episodes"][str(e)]["input_hash"] = "stale"
        for e in missing:
            del data["episodes"][str(e)]
        return json.dumps(data)

    # --- content -------------------------------------------------------------

    def test_structure(self):
        self.assertEqual(self.full["schema_version"], SCHEMA_VERSION)
        self.assertEqual(self.full["season"], SEASON)
        n = self.season.num_weeks
        self.assertEqual(list(self.full["episodes"]), [str(i + 1) for i in range(n)])
        for i in range(n):
            ep = self.full["episodes"][str(i + 1)]
            self.assertEqual(ep["input_hash"], episode_hash(self.season, i, CODE))
            self.assertGreater(ep["num_sols"], 0)
            self.assertEqual(set(ep["probs"]), {"index", "columns", "data"})

    def test_matches_old_df_for_probs(self):
        for ep in EPISODES:
            with self.subTest(ep=ep):
                sols = self.solver.solve(ep, True)
                expected = df_for_probs(SolutionSpace(self.solver, sols)).to_dict("split")
                actual = self.full["episodes"][str(ep + 1)]
                self.assertEqual(actual["num_sols"], len(sols))
                self.assertEqual(actual["probs"], expected)

    def test_true_solution_is_never_excluded(self):
        # every pair of the real solution must stay possible in every episode
        if not self.season.solution:
            self.skipTest("season has no known solution")
        for key, ep in self.full["episodes"].items():
            p = ep["probs"]
            rows = {name: i for i, name in enumerate(p["index"])}
            cols = {name: j for j, name in enumerate(p["columns"])}
            for pair in self.season.solution:
                l, r = self.season.decode_pair(pair)
                with self.subTest(episode=key, pair=(l, r)):
                    self.assertGreater(p["data"][rows[l]][cols[r]], 0)

    # --- caching -------------------------------------------------------------

    def test_cached_run_solves_nothing_and_keeps_the_file(self):
        out = self.fresh()
        with patch("scripts.export_site.get_solver") as get_solver_mock:
            run_export(out)
        get_solver_mock.assert_not_called()
        self.assertEqual(
            (out / "probs" / f"{SEASON}.json").read_text(encoding="utf-8"), self.full_text
        )

    def test_only_stale_and_missing_episodes_are_solved(self):
        out = self.fresh(self.tampered(stale=(3,), missing=(10,)))
        with patch("scripts.export_site.solution_chunks", wraps=solution_chunks) as spy:
            run_export(out)
        solved = sorted(call.args[1] for call in spy.call_args_list)  # 0-based
        self.assertEqual(solved, [2, 9])
        self.assertEqual(
            (out / "probs" / f"{SEASON}.json").read_text(encoding="utf-8"), self.full_text
        )

    def test_code_change_invalidates_the_cache(self):
        out = self.fresh()
        with self.assertRaises(SystemExit):
            run_export(out, code="some-other-code", max_solve=0)

    def test_unusable_cache_is_ignored(self):
        unusable = [
            "not json",
            json.dumps({"episodes": self.full["episodes"]}),  # old format, no schema_version
            json.dumps({**self.full, "schema_version": SCHEMA_VERSION + 1}),
        ]
        for text in unusable:
            with self.subTest(text=text[:20]), self.assertRaises(SystemExit):
                run_export(self.fresh(text), max_solve=0)

    # --- --max-solve guard ---------------------------------------------------

    def test_max_solve_aborts_before_solving_anything(self):
        out = self.fresh(self.tampered(stale=(3, 4, 5)))
        path = out / "probs" / f"{SEASON}.json"
        before = path.read_text(encoding="utf-8")
        with patch("scripts.export_site.get_solver") as get_solver_mock:
            with self.assertRaises(SystemExit) as cm:
                run_export(out, max_solve=2)
        get_solver_mock.assert_not_called()
        self.assertIn("--max-solve", str(cm.exception))
        self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_max_solve_allows_up_to_the_limit(self):
        out = self.fresh(self.tampered(missing=(10,)))
        run_export(out, max_solve=1)
        self.assertEqual(
            (out / "probs" / f"{SEASON}.json").read_text(encoding="utf-8"), self.full_text
        )

    def test_max_solve_zero_passes_with_a_complete_cache(self):
        run_export(self.fresh(), max_solve=0)


if __name__ == "__main__":
    unittest.main()
