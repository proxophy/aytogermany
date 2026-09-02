import pandas as pd

from ayto import AYTO



def check_nights(nights: list, lefts: list, rights: list) -> None:
    """Check if nights are fine"""

    # make sure all pairs are valid pairs and there are no duplicates
    night = 0
    for pairs, lights in nights:
        seated_lefts, seated_rights = [], []

        for l, r in pairs:
            if l in lefts and r in rights:
                # check that we don't have double seatings
                if l in seated_lefts or r in seated_rights:
                    raise ValueError(
                        f"{l} or {r} has already been seated  in night {night}"
                    )
                seated_lefts.append(l)
                seated_rights.append(r)
            elif l in rights and r in lefts:
                raise ValueError(
                    f"Pair {l} & {r} is in the wrong order in night {night}"
                )
            else:
                raise ValueError(f"{l} or {r} is neither in the list of women or men")
        if not 0 <= lights <= 10:
            raise ValueError(f"Number of lights not possible in night {night}")
        night += 1

def validate_season_args(
    lefts: list[str],
    rights: list[str],
    nights: list[tuple[list[tuple[str, str]], int]],
    matchboxes: dict[tuple[int, str, str], bool],
    dm: str | None,
):

    nummatches = max(len(lefts), len(rights))
    if len(lefts) != 10 or len(rights) != nummatches:
        raise ValueError(f"Not enough or too much women or men")

    # check nights
    check_nights(nights, lefts, rights)

    # matchboxes with results
    pairs = set()
    for (n, l, r), result in matchboxes.items():

        if (l, r) in pairs or (r, l) in pairs:
            raise AssertionError(f"{(l, r)} occurs more than once in matchboxes")
        if l in lefts and r in rights:
            pairs.add((l, r))
        elif l in rights and r in lefts:
            raise ValueError(f"Pair {l} & {r} is in the wrong order in matchboxes")
        else:
            raise ValueError(f"{l} or {r} is not a valid name")

    if dm and dm in lefts:
        raise ValueError(f"{dm} is part of lefts (the smaller gender group)")

    return

def read_data_from_excel(sn: str):
    dfcand = pd.read_excel(f"data/{sn}.xlsx", sheet_name="Candidates", header=0)
    lefts = dfcand["left"].dropna().tolist()
    rights = dfcand["right"].dropna().tolist()
    dmlist = dfcand["mm"].dropna().tolist() if "mm" in dfcand.columns else []
    if len(dmlist) > 1:
        raise ValueError(f"More than one DM in Candidates sheet for {sn}")
    dm = dmlist[0] if len(dmlist) == 1 else None
    if sn == "normalo2024":
        print("reading normalo2024")
        dm = dfcand["mm"].dropna().tolist()[0] if "mm" in dfcand.columns else None
        print("tm", dm)
    dfnights = pd.read_excel(f"data/{sn}.xlsx", sheet_name="Nights", header=0)
    nights = [
        (list(zip(dfnights.columns[:-1], row[:-1])), int(row[-1]))
        for row in dfnights.values.tolist()
    ]

    dfmatchboxes = pd.read_excel(f"data/{sn}.xlsx", sheet_name="Matchboxes", header=0)
    matchboxes = {
        (row[0], row[1], row[2]): row[3] for row in dfmatchboxes.values.tolist()
    }
    validate_season_args(lefts, rights, nights, matchboxes, dm)
    return lefts, rights, nights, matchboxes, dm, set()


if __name__ == "__main__":
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

    
