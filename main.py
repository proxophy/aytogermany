from ayto import *
from aytonormalo24 import AYTONormalo2024
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
    season: AYTO = AYTONormalo2024(*utils.read_data_normalo2024(sn))

    sn = "vip2025"
    season: AYTO = AYTO(*utils.read_data(sn))
    # options = {"end": 3,
    #     "includenight": True,  "verbose": False}
    # sols = find_solutions(season, options)
    # print(len(sols))
    # 2: 83141, 3: 2456
    # print(season.get_matchboxes({"end":3}))

    # sn_list = []
    # for sn in allseasons:
    #     print(sn)
    #     season: AYTO = AYTO(*utils.read_data(sn))
        
    #     options = {"end": 9,
    #             "includenight": True,  "verbose": False}
    #     solnum = {}
    #     for i in range(2, 10):
    #         options["end"] = i
    #         options["includenight"] = True
    #         sols1 = find_solutions(season, options, True)
    #         solnum[i] = len(sols1)
    #         print(f"i: {i} ({2*(i+1)+1}/{2*(i+1)+2})", len(sols1))
    #     sn_list.append(solnum)
    # df = pd.DataFrame(sn_list, index=allseasons)
    # df.to_csv("analytics/num_solutions.csv")   

    # options["end"] = 5
    # options["includenight"] = False
    # df = analysize_solutions(season, options)
    # print(df)

    # probs = matching_night_probs(season,8)
    # print(list(enumerate(probs)))
    # plot_df(df)
