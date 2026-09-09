from ayto import AYTO, AYTONormalo2024, AYTOVIP2023, AYTOVIP2025, AYTOVIP2026
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
    sn = "vip2021"
    options = {"end": 2, "includenight": True, "verbose": True}
    season: AYTO = AYTO(*utils.read_data_from_excel(sn))
    sols = find_solutions(season, options)
    print(len(sols))

    # season: AYTO = AYTO(*utils.read_data_from_excel("vip2024")) # 103
    # print(matching_night_probs(season, 3))
    nums = []
    # for i in range(2, 10):
    #     options["end"] = i
    #     sols = find_solutions(season, options)
    #     nums.append(len(sols))
    # print(nums)
    # print(sols[0])
    # print("number of solutions", len(sols))

    # df = analysize_solutions(season, options)
    # plot_df(df)

    exit()

    psol = {('Johannes', 'Janice'), ('Marwin', 'Michelle'), ('Germain', 'Emma'), ('Cansin', 'Julia'), ('Bennet', 'Christin'), ('Johannes', 'Marta'), ('Brian', 'Francesca'), ('Fabian', 'Alexandra'), ('Laurenz', 'Alexandra'), ('Robin', 'Joena'), ('Daymian', 'Zoe'), ('Raúl', 'Jenny')}

    # mathematical solution analysis stuff
    arr = []
    mins = []
    maxs = []
    for s in sols:
        _, m1, m2 = sol_probs(sols, s, options)
        mins.append(m1)
        maxs.append(m2)
    print(min(mins), max(mins))
    print(min(maxs), max(maxs))

    for s in sols:
        probs, m1, m2 = sol_probs(sols, s, options)
        if m1 <= min(mins):
            print("Found:", s, list(enumerate(probs)))
    
    
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
