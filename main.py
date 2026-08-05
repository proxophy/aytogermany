from ayto import *
from aytonormalo24 import AYTONormalo2024
from aytovip25 import AYTOVIP2025
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import utils


def plot_df(df):
    my_cmap = sns.light_palette((0.2,0.7,0.2), as_cmap=True)
    my_cmap.set_under((1,0.7,0.7))
    my_cmap.set_over((0.2,0.8,0.2))
    sns.heatmap(df, vmin=1e-5, vmax=100-1e-5, cmap=my_cmap, annot=True, fmt=".0f")
    print("df here",  df)
    plt.tight_layout()
    plt.show()


def index_to_episode(i):
    return 2*(i+1) + 1


if __name__ == "__main__":
    allseasons = ["normalo2020", "normalo2021", "normalo2022", "normalo2023", "normalo2025", "normalo2026"
                  "vip2021", "vip2022", "vip2023",  "vip2024"]

    # problem normalo 2024: includenight=False in 3rd episode includes Gerrit x (Tais, Mela)
    # when their from the end of the episode
    sn = "normalo2026"
    season: AYTO = AYTO(*utils.read_data(sn))

    options = {"end": 7,
               "includenight": True,  "verbose": True}
    
    
    # sol = {("Julian M", "Marla"), ("Noel", "Tonia"), ("Jeronymo", "Tiziana"), ("Chris", "Aurora"), ("Jerry", "Elena"),
    #        ("Julian S", "Adrianna"), ("Evi", "Linda"), ("Luke", "Ella"), ("Meji", "Michelle"), ("Ema", "Laura")}


    # 452 -> 439 -> 266
    # print(matching_night_probs(season,4))
    # res = season.parsol_possible(sol, options)
    # print(res)
    # df = analysize_solutions(season, options)
    sols = find_solutions(season, options)
    print("number of solutions", len(sols))
    print(sols[0])
    # print(df)
    # plot_df(df)


    # adf = pd.read_csv("analytics/num_solutions.csv", index_col=0)
    # print(adf)
