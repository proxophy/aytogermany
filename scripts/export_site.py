# pyright: standard
"""Export season results for the website.
 
Replaces scripts/export_season_results.py for the publishing pipeline.
Run from anywhere (the script switches to the repo root itself):
 
    uv run python -m scripts.export_site --out ../proxophy.github.io/static/ayto
    uv run python -m scripts.export_site --out site/static/ayto --season vip2026
    uv run python -m scripts.export_site --out site/static/ayto --force
 
Output (relative to --out):
 
    manifest.json              seasons, number of episodes, whether a solution is published
    probs/<season>.json        per episode: num_sols + pair probabilities ("split" format)
    data/<season>/<sheet>.csv  nights, matchboxes, (solution once it is known)
 
Re-running with unchanged inputs rewrites nothing. Every episode entry stores a hash
of everything its result depends on (solver code, season config, nights and matchboxes
up to that episode), so only new or changed episodes are solved again.
"""
 
import argparse
import dataclasses
import hashlib
import json
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
 
import numpy as np
import pandas as pd
 
from ayto import ALL_SEASONS, get_season, get_solver
from ayto.season import Season
 
SCHEMA_VERSION = 1
# Sheets of data/seasons/<season>.xlsx that the site reads as CSV.
SITE_SHEETS = ("nights", "matchboxes", "solution")
ROOT = Path(__file__).resolve().parent.parent
 
 
def dumps(obj: object) -> bytes:
    """Deterministic, diff-friendly JSON."""
    return (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
 
 
def write_if_changed(path: Path, data: bytes) -> bool:
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True
 
 
def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
 
 
def code_hash() -> str:
    """Hash of the solver package: any code change invalidates all cached episodes."""
    h = hashlib.sha256()
    for p in sorted((ROOT / "ayto").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]
 
 
def episode_hash(season: Any, i: int, code: str) -> str:
    """Hash of everything the result of solve(i, True) depends on."""
    static = {
        f.name: getattr(season, f.name)
        for f in dataclasses.fields(season)
        if f.name not in {"nights", "matchboxes", "solution"}
    }
    payload = {
        "code": code,
        "season": static,
        "nights": [[sorted(n.pairs), n.lights] for n in season.nights[: i + 1]],
        "matchboxes": [
            [w, list(p), r, bf] for w, p, r, bf in season.matchboxes if w <= i
        ],
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
 
 
def solution_chunks(solver: Any, episode: int) -> Iterator[list[int]]:
    """Yields the same solutions as solver.solve(episode, True), but one partial
    solution at a time, so the full list of solutions never exists in memory.
 
    Complete solutions generated from different partial solutions are disjoint
    (each solution determines its partial solution), so counts simply add up.
    """
    for g in solver.generate_partial_solutions(episode, True):
        yield solver.generate_complete_solutions(g, episode, True)
 
 
def pair_counts(
    chunks: Iterable[list[int]], n_bits: int, chunk_size: int = 100_000
) -> tuple[list[int], int]:
    """(count per pair bit, number of solutions); bit = left * R + right.
 
    Vectorised replacement for SolutionSpace.pairs_counter, which decodes every
    solution in pure Python. `chunks` are lists of solutions; they are processed
    one after the other and can be released afterwards.
    """
    words = (n_bits + 63) // 64
    mask = (1 << 64) - 1
    one = np.uint64(1)
    counts = [0] * n_bits
    total = 0
    for sols in chunks:
        total += len(sols)
        for start in range(0, len(sols), chunk_size):
            part = sols[start : start + chunk_size]
            arrays = [
                np.fromiter(
                    ((s >> (64 * w)) & mask for s in part), dtype=np.uint64, count=len(part)
                )
                for w in range(words)
            ]
            for b in range(n_bits):
                hits = (arrays[b // 64] >> np.uint64(b % 64)) & one
                counts[b] += int(np.count_nonzero(hits))
    return counts, total
 
 
def probs_table(
    season: Season, codec: Any, chunks: Iterable[list[int]]
) -> tuple[dict[str, Any], int]:
    """Same content as df_for_probs(...).to_dict("split"), in percent, and #solutions."""
    counts, n = pair_counts(chunks, codec.L * codec.R)
    if n == 0:
        raise ValueError("No solutions provided")
    lefts = sorted(season.lefts_names)
    rights = sorted(season.rights_names)
    li = {name: i for i, name in enumerate(season.lefts_names)}
    ri = {name: i for i, name in enumerate(season.rights_names)}
    data = [[counts[li[l] * codec.R + ri[r]] / n * 100 for r in rights] for l in lefts]
    return {"index": lefts, "columns": rights, "data": data}, n
 
 
def export_probs(
    sn: str, out: Path, force: bool, code: str, max_solve: int | None
) -> None:
    season = get_season(sn)
    path = out / "probs" / f"{sn}.json"
 
    old = None if force else load_json(path)
    cached = old["episodes"] if old and old.get("schema_version") == SCHEMA_VERSION else {}
 
    hashes = [episode_hash(season, i, code) for i in range(season.num_weeks)]
    stale = [
        i
        for i, h in enumerate(hashes)
        if (cached.get(str(i + 1)) or {}).get("input_hash") != h
    ]
    if max_solve is not None and len(stale) > max_solve:
        raise SystemExit(
            f"{sn}: {len(stale)} episode(s) would be solved ({[i + 1 for i in stale]}), "
            f"more than --max-solve {max_solve}. Did the solver code or the inputs change "
            "since the cached files were created? Use --force to solve anyway."
        )
 
    solver = None  # built lazily: not needed when everything is cached
    episodes: dict[str, Any] = {}
    solved = 0
    for i, h in enumerate(hashes):
        key = str(i + 1)
        if i not in stale:
            episodes[key] = cached[key]
            continue
 
        if solver is None:
            solver = get_solver(sn, True)
        t0 = time.perf_counter()
        probs, num_sols = probs_table(season, solver.codec, solution_chunks(solver, i))
        print(
            f"{sn} episode {i + 1}: {num_sols} solutions ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
        episodes[key] = {"input_hash": h, "num_sols": num_sols, "probs": probs}
        solved += 1
 
    data = {"schema_version": SCHEMA_VERSION, "season": sn, "episodes": episodes}
    changed = write_if_changed(path, dumps(data))
    print(f"{sn}: probs {'updated' if changed else 'unchanged'} ({solved} episodes solved)")
 
 
def export_csvs(sn: str, out: Path) -> None:
    sheets = pd.read_excel(f"data/seasons/{sn}.xlsx", sheet_name=None)
    by_name = {k.lower(): v for k, v in sheets.items()}
    for name in SITE_SHEETS:
        df = by_name.get(name)
        # No rows (e.g. solution of an ongoing season): publish nothing.
        if df is None or df.empty:
            continue
        text = df.to_csv(index=False, lineterminator="\n")
        write_if_changed(out / "data" / sn / f"{name}.csv", text.encode("utf-8"))
 
 
def write_manifest(out: Path) -> None:
    seasons = []
    for sn in ALL_SEASONS:
        data = load_json(out / "probs" / f"{sn}.json")
        if data is None:
            continue
        seasons.append(
            {
                "name": sn,
                "episodes": len(data["episodes"]),
                "has_solution": (out / "data" / sn / "solution.csv").exists(),
            }
        )
    manifest = {"schema_version": SCHEMA_VERSION, "seasons": seasons}
    write_if_changed(out / "manifest.json", dumps(manifest))
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # pyright: ignore[reportOptionalMemberAccess]
    parser.add_argument("--out", type=Path, required=True, help="e.g. <site repo>/static/ayto")
    parser.add_argument(
        "--season", nargs="*", default=None, help="seasons to export (default: all)"
    )
    parser.add_argument("--force", action="store_true", help="ignore cached episodes")
    parser.add_argument(
        "--max-solve",
        type=int,
        default=None,
        metavar="N",
        help="abort a season if more than N episodes would have to be solved "
        "(use in CI so a cache miss fails fast instead of recomputing everything)",
    )
    args = parser.parse_args()
 
    seasons: list[str] = args.season or list(ALL_SEASONS)
    unknown = sorted(set(seasons) - set(ALL_SEASONS))
    if unknown:
        parser.error(f"unknown season(s): {', '.join(unknown)}")
 
    out = args.out.resolve()  # resolve before changing directory
    os.chdir(ROOT)  # get_season() reads data/... relative to the repo root
 
    code = code_hash()
    for sn in seasons:
        export_probs(sn, out, args.force, code, args.max_solve)
        export_csvs(sn, out)
    write_manifest(out)
 
 
if __name__ == "__main__":
    main()
 