# pyright: standard

import argparse

from ayto import (
    Season,
    Solution,
    Solver,
    analyze_solutions,
    get_season,
    get_solver,
    plot_probs,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("season", type=str, default="vip2026")
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--analyze", action="store_true", default=False)
    args = parser.parse_args()
    sn: str = args.season
    end: int = args.end

    import time

    season: Season = get_season(sn)
    sol: Solution = season.solution
    solver: Solver = get_solver(sn)

    s = time.time()
    sols = solver.solve(end)
    e = time.time()

    print(f"Time taken: {(e - s):.2f} seconds")

    if args.analyze:
        analyze_solutions(sols)
    else:
        print(f"sn = {sn}, end = {end}: #solutions = {len(sols)}")
    
    if args.plot:
        plot_probs(sols)
    
    # plot_probs(sols)
