from collections import Counter

from ayto import Solution, Solver, get_season

# pyright: standard


def iterate_through(solver, start=3):
    amounts = []
    for end in range(start, 10):
        sols = solver.solve(end, True)
        amounts.append(len(sols))
    return amounts


def check_solution_possible(sols, solver: Solver, end):
    reasons = []
    for s in sols:
        p = solver.find_violation(s, end, True)
        if p is not None:
            reasons.append(p)
            # print(s, p["reason"], p.get("detail", ""))
    return Counter(reasons)


def findpsol(sol, solver: Solver, end):
    psols = solver.generate_partial_solutions(end, True)
    psol = Solution()
    for p in psols:
        if sol in solver.generate_complete_solutions(p, end, True):
            return p
    return psol


if __name__ == "__main__":
    # uv run python -m scripts.debug_season
    sn = get_season("normalo2024")
    print(sn.get_matchboxes(3, True))