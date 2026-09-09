import time
import tqdm

from .ayto import AYTO, PartialSol, CompleteSol
from .utils import time_it


def find_solutions_slow(season: AYTO, options: dict) -> list[CompleteSol]:
    solutions = season.generate_complete_solutions(set(), options)
    print(f"Generated solutions: {len(solutions)}")
    solutions = list(
        filter(lambda s: season.partialsol_possible(s, options), solutions)
    )

    return solutions


@time_it
def find_solutions(season: AYTO, options: dict) -> list[CompleteSol]:
    start = time.time()
    times = []
    verbose: bool = options.get("verbose", False)

    merged_partialsols = season.generate_partialsols(options)

    times.append(time.time() - start)
    start = time.time()
    if verbose:
        print(f"generate_partialsols done after {times[0]}")

    solutions_unfiltered: list[CompleteSol] = []
    for g in tqdm.tqdm(merged_partialsols):
        sols_g = season.generate_complete_solutions(g, options)
        solutions_unfiltered += sols_g

    times.append(time.time() - start)
    start = time.time()
    if verbose:
        print(f"generate_complete_solutions done after {times[1]}")

    # options.update({"checknights": True})
    solutions = list(
        filter(lambda s: season.partialsol_possible(s, options), solutions_unfiltered)
    )
    if len(solutions_unfiltered) - len(solutions) > 0:
        print("len(solutions_unfiltered) - len(solutions) > 0")

    times.append(time.time() - start)
    if verbose:
        print(f"Generating partialsols: {times[0]:0.2f} s")
        print(f"Generating solutions: {times[1]:0.2f} s")
        print(f"Filtering solutions: {times[2]:0.2f} s")
        print(
            f"Before and after filtering: {len(solutions_unfiltered)} {len(solutions)}"
        )

    return solutions

