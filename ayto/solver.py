import time
import tqdm
import itertools

from .ayto import Solver
from .utils import time_it
from .models import Solution, Pair

@time_it
def find_solutions_slow(solver: Solver, end: int) -> list[Solution]:
    def zip_product(clefts, ordering):
        return frozenset(p for p in zip(clefts, ordering))
    perms = itertools.permutations( solver.season.rights)
    parsols = [Solution(zip_product(solver.season.lefts, perm)) for perm in perms ]
    print("parsols generated")
    sols = set()
    for p in tqdm.tqdm(parsols):
        sols.update(solver.generate_complete_solutions(p, end))
    # season.generate_complete_solutions(Solution(), end)
    # print(f"Generated solutions: {len(solutions)}")
    # solutions = list(filter(lambda s: season.solution_possible(s, end), solutions)["res"])

    return list(sols)


@time_it
def find_solutions(solver: Solver, end: int) -> tuple[list[Solution], list]:
    start = time.time()
    times = []

    merged_partialsols = solver.generate_partial_solutions(end)
    times.append(("generate_partial_solutions", time.time() - start))
    start = time.time()

    solutions_unfiltered: list[Solution] = []
    

    for g in tqdm.tqdm(merged_partialsols):
        sols_g = solver.generate_complete_solutions(g, end)
        solutions_unfiltered += sols_g
    print("solutions_unfiltered", len(solutions_unfiltered))

    times.append(("generate_complete_solutions", time.time() - start))
    start = time.time()

    # options.update({"checknights": True})
    solutions = list(
        filter(lambda s: solver.solution_possible(s, end)["res"], solutions_unfiltered)
    )
    if len(solutions_unfiltered) - len(solutions) > 0:
        print("len(solutions_unfiltered) - len(solutions) > 0")

    times.append(("filtering with solution_possible", time.time() - start))
    return solutions, times
