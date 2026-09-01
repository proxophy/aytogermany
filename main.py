from ayto import *
from aytonormalo24 import AYTONormalo2024
from aytovip25 import AYTOVIP2025
from aytovip23 import AYTOVIP2023
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
    allseasons = ["normalo2020", "normalo2021", "normalo2022", "normalo2023", "normalo2024", "normalo2025", "normalo2026",
                  "vip2021", "vip2022", "vip2023",  "vip2024", "vip2025", "vip2026"]

    # problem normalo 2024: includenight=False in 3rd episode includes Gerrit x (Tais, Mela)
    # when their from the end of the episode
    import json
    

    sn = "normalo2024"
    with open(f"data/{sn}.json", "r") as f:
        jsondata: dict = json.loads(f.read())
    
    season: AYTO = AYTONormalo2024(*utils.read_data_from_excel(sn))
    # print(season.nights[-1])
    options = {"end": 8,
               "includenight": False,  "verbose": False}
    # print("res",  res)
    # utils.seasontoexcel(season,sn)

    # sols = find_solutions(season, options)
    # print("number of solutions", len(sols)) # 115 
    # print(sols[0])
    # df = analysize_solutions(season, options)
    # exit()
   
    # allseasons = ["vip2021", "vip2022"]
    for sn in allseasons[:-2]:
        print(sn)
        if sn == "normalo2024":
            args = utils.read_data_normalo2024(sn)
            season = AYTONormalo2024(*args)
        elif sn == "vip2025":
            args = utils.read_data(sn)
            season = AYTOVIP2025(*args)
        elif sn == "vip2023":
            args = utils.read_data(sn)
            season = AYTOVIP2023(*args)
        else:
            args = utils.read_data(sn)
            season = AYTO(*args)
        utils.seasontoexcel(season,sn)
        # options = {"end": 4,
        #                 "includenight": True,  "verbose": True}
        # sols = find_solutions(season, options)
       
    # adf = pd.read_csv("analytics/num_solutions.csv", index_col=0)
    # print(adf)
