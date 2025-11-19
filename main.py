from ayto import *
from aytonormalo24 import AYTONormalo2024
from aytovip25 import AYTOVIP2025
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import utils


def plot_df(df):
    sns.heatmap(df, vmin=0, vmax=100, cmap="BuGn", annot=True, fmt=".0f")
    print("df here",  df)
    plt.tight_layout()
    plt.show()


def index_to_episode(i):
    return 2*(i+1) + 1


if __name__ == "__main__":
    allseasons = ["normalo2020", "normalo2021", "normalo2022", "normalo2023", "normalo2025",
                  "vip2021", "vip2022", "vip2023",  "vip2024"]

    # problem normalo 2024: includenight=False in 3rd episode includes Gerrit x (Tais, Mela)
    # when their from the end of the episode
    sn = "normalo2024"
    seasonnormalo2024: AYTO = AYTONormalo2024(*utils.read_data_normalo2024(sn))

    sn = "vip2025"
    season: AYTO = AYTOVIP2025(*utils.read_data(sn))

    options = {"end": 10,
               "includenight": True,  "verbose": True}
    
    # arr = []
    # for i in range(2, 10):
    #     sols = find_solutions(seasonnormalo2024, {"end": i, "verbose": True})
    #     arr.append((i, len(sols)))
    # print(arr)

    # 452 -> 439 -> 266
    print(matching_night_probs(season,10))
    df = analysize_solutions(season, options)
    # print(df)
    plot_df(df)

    # Nach Jimy matchbox: 7531, nach Matchingnight 452

    # adf = pd.read_csv("analytics/num_solutions.csv", index_col=0)
    # print(adf)
