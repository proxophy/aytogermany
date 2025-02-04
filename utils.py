import json
import pandas as pd
from typing import List, Tuple, Dict, Set, Union, Optional
from ayto import AYTO

from jsonschema import validate

schema = {
    "type": "object",
    "additionalProperties": False,
    "required": ["men", "women", "matchboxes"],
    "properties": {
            "men": {"type": "array",
                    "items": {"type": "string"},
                    "minItems": 10,
                    "maxItems": 11,
                    "uniqueItems": True
                    },
            "women": {"type": "array",
                      "items": {"type": "string"},
                      "minItems": 10,
                      "maxItems": 12,
                      "uniqueItems": True},
            "matchboxes": {
                "type": "array",
                "items": {
                    "type": "array",
                    "prefixItems": [
                        {"type": "integer", "minimum": 0, "maximum": 10},
                        {"type": "string"},
                        {"type": "string"},
                        {"type": "boolean"}
                    ],
                    "maxItems": 4,
                    "minItems": 4
                },
                "maxItems": 12,
            },
        "dm": {"type": "string"},
        "bonights": {"type": "array",
                     "items": {"type": "integer",
                               "minimum": 0,
                               "maximum": 9},
                     "maxItems": 10, },
        "cancelled": {"type": "integer",
                      "minimum": 0,
                      "maximum": 9},
        "dmtuple": {
                "type": "array",
                "prefixItems": [
                        {"type": "string"},
                        {"type": "string"},
                ],
                "maxItems": 2,
                "minItems": 2
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
                    "minItems": 2
                }
        }
    }
}


def check_nights(nights: list, lefts: list, rights: list) -> bool:
    '''Check if nights are fine'''

    # make sure all pairs are valid pairs and there are no duplicates
    for pairs, lights in nights:
        seated_lefts, seated_rights = [], []

        for l, r in pairs:
            if l in lefts and r in rights:
                # check that we don't have double seatings
                if l in seated_lefts or r in seated_rights:
                    print(f"{l} or {r} has already been seated")
                    return False
                seated_lefts.append(l)
                seated_rights.append(r)
            elif l in rights and r in lefts:
                # check that we don't have double seatings
                if l in seated_rights or r in seated_lefts:
                    print(f"{l} or {r} has already been seated")
                    return False
                seated_lefts.append(r)
                seated_rights.append(l)
            else:
                print(f"{l} or {r} is neither in the list of women or men")
                return False
        if not 0 <= lights <= 10:
            print(f"Number of lights not possible")
            return False
    return True


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
        f"data/{sn}.xlsx", sheet_name="Nights",
        header=0, index_col=0)
    leftsh = nightsdf.columns.values.tolist()
    nights = [(list(zip(leftsh, row[:-1])), row[-1])
              for row in nightsdf.values.tolist()]
    # check nights
    assert check_nights(nights, lefts, rights)

    # matchboxes with results
    matchboxes = {}
    boxesepisodes = []
    for n, l, r, result in jsondata["matchboxes"]:
        if (l, r) in matchboxes or (r, l) in matchboxes:
            raise AssertionError(
                f"{(l, r)} occurs more than once in matchboxes")

        if l in lefts and r in rights:
            matchboxes[(l, r)] = result
        elif l in rights and r in lefts:
            matchboxes[(r, l)] = result
        else:
            print(l, r)
            print(jsondata["matchboxes"])
            raise AssertionError(f"{l} or {r} is not a valid name")
        boxesepisodes.append(n)

    # nights with blackout
    bonights = jsondata["bonights"]

    # do we know who is the second/third match to someone
    # besides normalo2023, yes
    dm = jsondata.get("dm", None)
    assert dm is None or dm in rights
    # do we know which two share the same match
    # besides vip2023, no
    dmtuple = jsondata.get("dmtuple", None)
    assert dmtuple is None or (dmtuple[0] in rights and dmtuple[1] in rights), \
        f"{dmtuple} may have spelling error"

    # matching night that was cancelled
    cancellednight = jsondata.get("cancelled", -1)
    assert -1 <= 0 <= 9, f"Invalid value for cancelled night"

    # mapping from matchboxes to episode
    # boxesepisodes = jsondata.get("boxesepisode", list(range(10)))
    # assert len(boxesepisodes) == len(
    #     matchboxes), f"{len(boxesepisodes), len(matchboxes)}"

    # solution
    solution = {(l, r) for l, r in jsondata.get("solution", [])}

    return lefts, rights, nights, matchboxes, bonights, dm, dmtuple, cancellednight, boxesepisodes, solution


def read_data_normalo2024(sn: str):
    with open(f"data/{sn}.json", "r") as f:
        jsondata: dict = json.loads(f.read())
    lefts, rights, nights, matchboxes, bonights, _, _, cancellednight, boxesepisodes, solution = read_data(
        sn)
    tm = jsondata.get("tm", None)
    assert tm in rights
    return lefts, rights, nights, tm, matchboxes, bonights, cancellednight, boxesepisodes, solution


if __name__ == "__main__":
    allseasons = ["normalo2020", "normalo2021", "normalo2022", "normalo2023",
                  "vip2021", "vip2022", "vip2023",
                  "vip2024"]
    season = "normalo2022"
