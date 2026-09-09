from ayto import AYTO, Solution
from ayto import find_solutions, sol_probs
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
    options = {"end": 5, "includenight": True, "verbose": False}
    lefts, rights, nights, matchboxes, dm, solution = utils.read_data_from_excel(sn)
    # print(matchboxes.get_perfect_matches(7))
    season: AYTO = AYTO(*utils.read_data_from_excel(sn))
    from ayto import Pair

    
    sols = find_solutions(season, options)
    # print("sol in generated", sol in  sols)
    print(len(sols))

   
