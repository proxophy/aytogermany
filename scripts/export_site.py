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
from pathlib import Path
from typing import Any

import pandas as pd

from ayto import (
    ALL_SEASONS,
    Season,
    SolutionSpace,
    df_for_probs,
    get_season,
    get_solver,
)

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


def episode_hash(season: Season, i: int, code: str) -> str:
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


def export_probs(sn: str, out: Path, force: bool, code: str) -> None:
    season = get_season(sn)
    path = out / "probs" / f"{sn}.json"

    old = None if force else load_json(path)
    cached = old["episodes"] if old and old.get("schema_version") == SCHEMA_VERSION else {}

    solver = None  # built lazily: not needed when everything is cached
    episodes: dict[str, Any] = {}
    solved = 0
    for i in range(season.num_weeks):
        key = str(i + 1)
        h = episode_hash(season, i, code)
        entry = cached.get(key)
        if entry is not None and entry.get("input_hash") == h:
            episodes[key] = entry
            continue

        if solver is None:
            solver = get_solver(sn, True)
        sols = solver.solve(i, True)
        print(f"{sn} episode {i + 1}: {len(sols)} solutions")
        df = df_for_probs(SolutionSpace(solver, sols))
        episodes[key] = {
            "input_hash": h,
            "num_sols": len(sols),
            "probs": df.to_dict("split"),  # keys: index, columns, data
        }
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
    args = parser.parse_args()

    seasons: list[str] = args.season or list(ALL_SEASONS)
    unknown = sorted(set(seasons) - set(ALL_SEASONS))
    if unknown:
        parser.error(f"unknown season(s): {', '.join(unknown)}")

    out = args.out.resolve()  # resolve before changing directory
    os.chdir(ROOT)  # get_season() reads data/... relative to the repo root

    code = code_hash()
    for sn in seasons:
        export_probs(sn, out, args.force, code)
        export_csvs(sn, out)
    write_manifest(out)


if __name__ == "__main__":
    main()
