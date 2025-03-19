from ayto import *
import seaborn as sns
import matplotlib.pyplot as plt


def plot_df(df):
    sns.heatmap(df, vmin=0, vmax=100, cmap="BuGn", annot=True, fmt=".0f")
    print("df here",  df)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    allseasons = ["normalo2020", "normalo2021", "normalo2022", "normalo2023", "normalo2025"
                  "vip2021", "vip2022", "vip2023",  "vip2024"]

    import utils

    sn = "vip2023"

    season: AYTO = AYTO(*utils.read_data(sn))
    options = {"end": 3,
               "includenight": True,  "verbose": False}

    # df = analysize_solutions(season, options)
    # print(df)

    sols1 = find_solutions(season, options, True)
    print(len(sols1))

    # probs = matching_night_probs(season,5)
    # print(list(enumerate(probs)))
    # plot_df(df)
