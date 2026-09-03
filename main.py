from ayto import *
from aytonormalo24 import AYTONormalo2024
from aytovip25 import AYTOVIP2025
from aytovip23 import AYTOVIP2023
from aytovip26 import AYTOVIP2026
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import utils


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
    options = {"end": 3, "includenight": True, "verbose": False}
    season: AYTO = AYTOVIP2026(*utils.read_data_from_excel(sn))
    season: AYTO = AYTO(*utils.read_data_from_excel("vip2024")) # 103
    
    sols = find_solutions(season, options)

    # df = analysize_solutions(season, options)
   
    print("number of solutions", len(sols))
    # plot_df(df)
    # print(sols[0])

    # res = season.partialsol_possible(tsol, options)
    # print("Result:", res)

    allseasons = [
        "normalo2020",
        "normalo2021",
        "normalo2022",
        "normalo2023",
        "normalo2024",
        "normalo2025",
        "normalo2026",
        "vip2021",
        "vip2022",
        "vip2023",
        "vip2024",
        "vip2025",
        "vip2026",
    ]
    exit()
    d = {}
    for sn in allseasons[:-1]:
        print(sn)
        if sn == "normalo2024":
            args = utils.read_data_from_excel(sn)
            season = AYTONormalo2024(*args)
        elif sn == "vip2025":
            args = utils.read_data_from_excel(sn)
            season = AYTOVIP2025(*args)
        elif sn == "vip2023":
            args = utils.read_data_from_excel(sn)
            season = AYTOVIP2023(*args)
        else:
            args = utils.read_data_from_excel(sn)
            season = AYTO(*args)
        sols = find_solutions(season, options)
        d[sn] = len(sols)
    print(d)
