from ayto import Solver, VIP2026Solver
from ayto.analysis import SolutionSpace,  analyze_solutions
from ayto.models import Season, Solution, Pair
from ayto.solver import find_solutions
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import ayto.utils as utils


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


if __name__ == "__main__":
    sn = "vip2026"
    options = {"end": 8, "includenight": True, "verbose": True}
    lefts, rights, nights, matchboxes, dm, solution = utils.read_data_from_excel(sn)
    # print(matchboxes.get_perfect_matches(7))
    psol = Solution(frozenset((Pair('Bennet', 'Francesca'), Pair('Fabian', 'Christin'), Pair('Johannes', 'Marta'), Pair('Cansin', 'Zoe'), Pair('Germain', 'Alexandra'), Pair('Raúl', 'Michelle'), Pair('Johannes', 'Janice'), Pair('Robin', 'Joena'), Pair('Daymian', 'Jenny'), Pair('Brian', 'Julia'), Pair('Laurenz', 'Alexandra'), Pair('Marwin', 'Emma'))))
    
    from ayto import Pair
    season: Season = Season(lefts, rights, nights, matchboxes, solution=solution, ldm=True)
    solver: Solver = VIP2026Solver(season)
    print(solver.sm)
    
    res = solver.solution_possible(psol, 8)
    print(res)
    

    # df = pd.Series(probs).unstack(fill_value=0)
   

   
