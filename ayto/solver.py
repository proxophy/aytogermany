import time
import tqdm

from .ayto import Solver
from .utils import time_it
from .models import Solution


def find_solutions_slow(season: Solver, end: int) -> list[Solution]:
    solutions = season.generate_complete_solutions(Solution(), end)
    print(f"Generated solutions: {len(solutions)}")
    solutions = list(filter(lambda s: season.solution_possible(s, end), solutions))

    return solutions


@time_it
def find_solutions(solver: Solver, end: int) -> list[Solution]:
    start = time.time()
    times = []

    merged_partialsols = solver.generate_partial_solutions(end)
    print("merged_partialsols",len( merged_partialsols))

    times.append(time.time() - start)
    start = time.time()

    solutions_unfiltered: list[Solution] = []
    

    for g in tqdm.tqdm(merged_partialsols):
        sols_g = solver.generate_complete_solutions(g, end)
        solutions_unfiltered += sols_g
    print("solutions_unfiltered", len(solutions_unfiltered))

    times.append(time.time() - start)
    start = time.time()

    # options.update({"checknights": True})
    solutions = list(
        filter(lambda s: solver.solution_possible(s, end), solutions_unfiltered)
    )
    print(solutions_unfiltered[0])
    if len(solutions_unfiltered) - len(solutions) > 0:
        print("len(solutions_unfiltered) - len(solutions) > 0")

    times.append(time.time() - start)

    return solutions
