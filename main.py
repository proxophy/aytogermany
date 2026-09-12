from ayto import Solver, VIP2026Solver, VIP2025Solver
from ayto.analysis import SolutionSpace, analyze_solutions
from ayto.models import Season, Solution, Pair
from ayto.solver import find_solutions, find_solutions_slow
from ayto.utils import get_season, get_solver
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_df(df):
    my_cmap = sns.light_palette((0.2, 0.7, 0.2), as_cmap=True)
    my_cmap.set_under((1, 0.7, 0.7))
    my_cmap.set_over((0.2, 0.8, 0.2))
    sns.heatmap(df, vmin=1e-5, vmax=100 - 1e-5, cmap=my_cmap, annot=True, fmt=".0f")
    plt.tight_layout()
    plt.show()


def index_to_episode(i):
    return 2 * (i + 1) + 1


def appendsol(sn: str, tsol: set[tuple[str, str]]):
    plefts, prights = zip(*tsol)
    dfsol = pd.DataFrame(data={"left": plefts, "right": prights})
    dfsol = dfsol.sort_values(by=["left", "right"]).reset_index(drop=True)
    print(dfsol)
    with pd.ExcelWriter(
        f"data/{sn}.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        dfsol.to_excel(writer, sheet_name="Solution", index=False)


def comparetoao(sols, end):
    with open("aodata.txt", "r") as f:
        lines = map(eval, f.readlines())

    def parse_sol(ld):
        pairs = set()
        for l, rs in ld.items():
            for r in rs:
                pairs.add((l, r))
        return Solution(frozenset(pairs))

    solsao = list(map(parse_sol, lines))

    print("lengths", len(solsao), len(sols))
    count = count2 = 0

    for s in solsao:
        spd = solver.solution_possible(s, end)
        if not spd["res"]:
            # print(spd["reason"])  
            count += 1
        elif s not in sols:
            count2 += 1
            print("s not in sols")
    print("count", count, count2)
    count = count2 = 0
    for s in sols:
        if s not in solsao:
            # print(s)
            count += 1
            rpos = solver.solution_possible(s, end )
            if not rpos["res"]:
                print(rpos["reason"], rpos.get("detail",""))
                count2 += 1
            else:
                sol = s
                print(s, rpos["res"])

    print("count", count, count2)


if __name__ == "__main__":
    sn = "normalo2026"

    season: Season = get_season(sn)
    sol: Solution = season.solution  # type: ignore
    solver: Solver = get_solver(sn)
    # print(solver.no_match("Cecilia", "Felix", 6))
    end = 7
    # sols, times = find_solutions(solver, end)
    # print(len(sols))

    
    amounts = []
    for end in range(2, 10):
        sols, times = find_solutions(solver, end)
        # sols = times = []
        print(len(sols))
        print(times)
        print(solver.times)
        amounts.append(len(sols))
    print(amounts)

    # comparetoao(sols, 6)
    exit()

    s = Solution(
        frozenset(
            (
                ("Manuel", "Kathleen"),
                ("Francesco", "Jules"),
                ("Alex", "Sarah"),
                ("Diogo", "Finnja"),
                ("Salvatore", "Jacky"),
                ("Francesco", "Vanessa"),
                ("Tommy", "Jill"),
                ("Eugen", "Walentina"),
                ("Danilo", "Melina"),
                ("Jamy", "Steffi"),
                ("Josua", "Aurelia"),
            )
        )
    )

    # res = solver.solution_possible(s, end + 1)
    # print("res", res)
    # print("no_match", solver.no_match("Francesco", "Vanessa", end))
    # solspace = SolutionSpace(sols)
    # probs = solspace.get_pair_probs_dict()

    # df = pd.Series(probs).unstack(fill_value=0)
    # print(df)
    # analyze_solutions(sols)
    # plot_df(df)
