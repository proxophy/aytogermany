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
    # sn = "normalo2024"
    # season: AYTO = AYTONormalo2024(*utils.read_data_normalo2024(sn))

    sn = "vip2025"
    season: AYTO = AYTOVIP2025(*utils.read_data(sn))
    season2: AYTO = AYTOVIP2025(*utils.read_data(sn))

    options = {"end": 10,
               "includenight": True,  "verbose": True}

    test1 = {('Henna', 'Jimi'), ('Beverly', 'Sidar'), ('Henna', 'Lennert'),
             ('Antonia', 'Olli'), ('Elli', 'Xander'),
             ('Viki', 'Leandro'), ('Nelly', 'Calvin O.'), ('Viki', 'Jonny')}
    #
    test2 = {('Viki', 'Kevin'), # ('Viki', 'Jonny'),
             ('Henna', 'Olli'),  ('Henna', 'Jimi'), 
             ('Nelly', 'Calvin O.'), ('Elli', 'Xander'), ('Beverly', 'Calvin S.')
              }

    test3 = {('Beverly', 'Nico'), ('Nelly', 'Calvin O.'), ('Joanna', 'Rob'),
              ('Elli', 'Xander'), ('Sandra', 'Olli'), ('Hati', 'Leandro'), 
              ('Henna', 'Jimi'), ('Antonia', 'Jonny'), ('Ariel', 'Calvin S.'), ('Viki', 'Kevin')}
    # gsols = season.generate_parsols(options)
    # gsols2 = season.generate_parsols(options)
    # sols = season.generate_complete_solutions(test3, options)
    # print("THIS: ", len(sols))
    # for s in sols:
    #     # print(s)
    #     if season.parsol_possible(s,options):
    #         print("parsol possible")
    #     if ("Henna", "Lennert") in s and ("Henna", "Sidar") in s:
    #         print(s, season.parsol_possible(s, options))
    # print("THIS 2", season.parsol_possible(test3, options))

    # for ps in gsols:
    #     plefts, prights, mminsol = season.get_parsol_leftrights(ps)
    #     if not season.parsol_possible(ps,options):
    #         print("TARGET")
    #     if abs(len(plefts)-len(prights)) == 2 and len(ps) == 7:
    #         print(ps)

    df = analysize_solutions(season, options)
    print(df)
    plot_df(df)
