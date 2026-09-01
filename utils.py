import json
import pandas as pd

from jsonschema import validate

from ayto import AYTO



schema = {
    "type": "object",
    "additionalProperties": False,
    "required": ["men", "women", "matchboxes"],
    "properties": {
        "men": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 10,
            "maxItems": 12,
            "uniqueItems": True,
        },
        "women": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 10,
            "maxItems": 12,
            "uniqueItems": True,
        },
        "matchboxes": {
            "type": "array",
            "items": {
                "type": "array",
                "prefixItems": [
                    {"type": "integer", "minimum": 0, "maximum": 10},
                    {"type": "string"},
                    {"type": "string"},
                    {"type": "boolean"},
                ],
                "maxItems": 4,
                "minItems": 4,
            },
            "maxItems": 12,
        },
        "bonights": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": 9},
            "maxItems": 10,
        },
        "dm": {"type": "string"},
        "tm": {"type": "string"},
        "cancelled": {"type": "integer", "minimum": 0, "maximum": 9},
        "dmtuple": {
            "type": "array",
            "prefixItems": [
                {"type": "string"},
                {"type": "string"},
            ],
            "maxItems": 2,
            "minItems": 2,
        },
        "solution": {
            "type": "array",
            "items": {
                "type": "array",
                "prefixItems": [
                    {"type": "string"},
                    {"type": "string"},
                ],
                "maxItems": 2,
                "minItems": 2,
            },
        },
    },
}


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
                    raise ValueError(f"{l} or {r} has already been seated  in night {night}")
                seated_lefts.append(l)
                seated_rights.append(r)
            elif l in rights and r in lefts:
                # check that we don't have double seatings
                raise ValueError(f"Pair {l} & {r} is in the wrong order in night {night}")
            else:
                raise ValueError(f"{l} or {r} is neither in the list of women or men")
        if not 0 <= lights <= 10:
            raise ValueError(f"Number of lights not possible in night {night}")
        night += 1


def read_data(sn: str):
    with open(f"data/{sn}.json", "r") as f:
        jsondata: dict = json.loads(f.read())

    validate(jsondata, schema)

    women = jsondata["women"]
    men = jsondata["men"]
    nummatches = max(len(women), len(men))

    # set lefts and rights
    if len(women) == nummatches and len(men) == 10:
        lefts, rights = men, women
    elif len(women) == 10 and len(men) == nummatches:
        lefts, rights = women, men
    else:
        raise ValueError(f"Not enough or too much women or men")

    # reading nights: seatings and lights for nights
    nightsdf = pd.read_excel(
        f"data/{sn}.xlsx", sheet_name="Nights", header=0, index_col=0
    )
    leftsh = nightsdf.columns.values.tolist()
    nights = [
        (list(zip(leftsh, row[:-1])), int(row[-1])) for row in nightsdf.values.tolist()
    ]
    # check nights
    check_nights(nights, lefts, rights)

    # matchboxes with results
    matchboxes = {}
    pairs = set()
    for n, l, r, result in jsondata["matchboxes"]:
        if (l, r) in pairs or (r, l) in pairs:
            raise AssertionError(f"{(l, r)} occurs more than once in matchboxes")
        if l in lefts and r in rights:
            matchboxes[(n, l, r)] = result
            pairs.add((l, r))
        elif l in rights and r in lefts:
            matchboxes[(n, r, l)] = result
            pairs.add((r, l))
        else:
            print(l, r)
            print(jsondata["matchboxes"])
            raise AssertionError(f"{l} or {r} is not a valid name")

    # do we know who is the second/third match to someone
    # besides normalo2023, yes
    dm = jsondata.get("dm", None)
    assert dm is None or dm in rights
    # do we know which two share the same match
    # besides vip2023, no
    dmtuple = jsondata.get("dmtuple", None)
    assert dmtuple is None or (
        dmtuple[0] in rights and dmtuple[1] in rights
    ), f"{dmtuple} may have spelling error"

    # solution
    solution = {(l, r) for l, r in jsondata.get("solution", [])}

    return lefts, rights, nights, matchboxes, dm, solution

def validate_season_args(lefts: list[str],
                         rights: list[str], 
                         nights: list[tuple[list[tuple[str, str]], int]], 
                         matchboxes: dict[tuple[int, str, str], bool], 
                         dm: str | None):

    
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

def read_data_normalo2024(sn: str):
    with open(f"data/{sn}.json", "r") as f:
        jsondata: dict = json.loads(f.read())
    lefts, rights, nights, matchboxes, _, solution = read_data(sn)
    tm: str = jsondata.get("tm", "")
    assert tm in rights
    return lefts, rights, nights, matchboxes, tm, solution


def seasontoexcel(season: AYTO, sn: str):
    dfcand = pd.DataFrame(
        data={
            "left": season.lefts + [None] * (len(season.rights) - len(season.lefts)),
            "right": season.rights,
            "mm": [season.tm if sn == "normalo2024" else season.dm] + [None] * (len(season.rights) - 1), # type: ignore
        },
    )

    def nighttodict(night: tuple[list[tuple[str, str]], int]) -> dict:
        pairs, lights = night
        return {l: r for l, r in pairs} | {"lights": lights}

    dfnights = pd.DataFrame([nighttodict(night) for night in season.nights])
    dfmatchboxes = pd.DataFrame(
        data={
            "episode": season.boxesepisodes,
            "left": [l for l, r in season.matchboxes],
            "right": [r for l, r in season.matchboxes],
            "result": [season.matchboxes[p] for p in season.matchboxes],
        }
    )
    with pd.ExcelWriter(f"data/{sn}v2.xlsx", engine="xlsxwriter") as writer:
        dfcand.to_excel(writer, sheet_name="Candidates", index=False)
        dfnights.to_excel(writer, sheet_name="Nights", index=False)
        dfmatchboxes.to_excel(writer, sheet_name="Matchboxes", index=False)

        # formatting
        workbook = writer.book
        header_format = workbook.add_format(
            {"bold": True, "bg_color": "#C5C6C7", "align": "center"}
        )
        for sheet_name, df in [
            ("Candidates", dfcand),
            ("Nights", dfnights),
            ("Matchboxes", dfmatchboxes),
        ]:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column(0, len(df.columns) - 1, 10)
            for col_num, value in enumerate(df.columns):
                worksheet.write(0, col_num, value, header_format)


def read_data_from_excel(sn: str):
    dfcand = pd.read_excel(f"data/{sn}v2.xlsx", sheet_name="Candidates", header=0)
    lefts = dfcand["left"].dropna().tolist()
    rights = dfcand["right"].dropna().tolist()
    dmlist = dfcand["mm"].dropna().tolist() if "mm" in dfcand.columns else []
    if len(dmlist) > 1:
        raise ValueError(f"More than one DM in Candidates sheet for {sn}")
    dm = dmlist[0] if len(dmlist) == 1  else None
    if sn == "normalo2024":
        print("reading normalo2024")
        dm = dfcand["mm"].dropna().tolist()[0] if "mm" in dfcand.columns else None
        print("tm", dm)
    dfnights = pd.read_excel(f"data/{sn}v2.xlsx", sheet_name="Nights", header=0)
    nights = [
        (list(zip(dfnights.columns[:-1], row[:-1])), int(row[-1]))
        for row in dfnights.values.tolist()
    ]

    dfmatchboxes = pd.read_excel(f"data/{sn}v2.xlsx", sheet_name="Matchboxes", header=0)
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

    

    # sn = "vip2024"
    # df = pd.read_excel(f"data/{sn}.xlsx", sheet_name="Matchboxes", header=0)

    # args = read_data_from_excel(sn)
    # seasontoexcel(AYTO(*args), sn)
    # men = args[0]
    # women = args[1]
    # matchboxes = args[3]
    # sn = AYTO(*args)
    # print(sn.matchboxes)
