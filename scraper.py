import pandas as pd
import re
import json

pseudonyms_spellingerros = {"Katharina": "Kathi",
                            "Antonio": "Antonino", "Pharell": "Pharrell"}


class Season_Parser():
    def __init__(self,
                 seasonname,
                 mencast, womencast,
                 boxes,
                 nightguys, nightwomen) -> None:

        self.seasonname = seasonname
        self.men = parse_cast_table(mencast)
        self.women = parse_cast_table(womencast)
        print(self.men, len(self.men))
        print(self.women, len(self.women))
        assert (len(self.men) == 10 and len(self.women) == 11) or \
            (len(self.men) == 11 and len(self.women) == 10), \
            "One gender must have 11, the other 10"

        if len(self.men) < len(self.women):
            self.left = "Guys"
            self.nighttable = nightguys
        else:
            self.left = "Girls"
            self.nighttable = nightwomen

        self.matchboxes = self.parse_boxes(boxes)
        print(self.matchboxes)

        self.parse_nights()
        for night in self.nights:
            # continue
            print(night)

    def parse_boxes(self, boxes):
        def split_pair(pair: str):
            p1, p2 = pair.split(" & ")
            p1, p2 = clean_name(p1), clean_name(p2)
            return p1, p2

        couples = boxes["Couple"].to_list()
        results = boxes["Result"].to_list()

        matchboxes = {}
        for couple, result in zip(couples, results):
            if result == "No Match" or result == "Perfect Match":
                l, r = split_pair(couple)
                l, r = self.correct_pair(l, r)

                matchboxes[(l, r)] = result == "Perfect Match"

        return matchboxes

    def parse_nights(self):
        nightsdf = self.nighttable["Matching Night"]
        lefts = self.nighttable[self.left][self.left].to_list()[:-1]

        self.nights = []
        self.bonights = []
        self.cancellednight = -1
        for i in nightsdf.columns:
            rights = nightsdf[i].to_list()[:-1]
            lights = nightsdf[i].to_list()[-1]

            if 'canceled' in rights[0]:
                self.cancellednight = int(i) - 1
                continue

            pairs = []
            print(i)
            for l, r in zip(lefts, rights):
                if r == "Not in the Game" or not isinstance(r, str):
                    continue
                print(l, r)
                l, r = self.correct_pair(l, r)
                pairs.append((l, r))

            assert len(pairs) == 10

            if "BLACK" in lights:
                self.bonights.append(int(i)-1)
                lights = lights[:1]

            lights = int(lights)

            self.nights.append((pairs, lights))

        print(self.bonights)
        return nights

    def correct_pair(self, p1, p2):
        if self.left == "Guys":
            lefts = self.men
            rights = self.women
        else:
            lefts = self.women
            rights = self.men

        p1, p2 = clean_name(p1), clean_name(p2)

        if p1 not in lefts and p1 not in rights:
            raise ValueError(f"Issue with the name {p1}")
        elif p2 not in rights and p2 not in lefts:
            raise ValueError(f"Issue with the name {p2}")

        # swap if needed
        if p1 in rights and p1 in lefts:
            # swap pair
            return p2, p1

        # everything is fine
        return p1, p2

    def write_json(self):
        mbtrans = [[[l, r], self.matchboxes[l, r]]
                   for (l, r) in self.matchboxes]
        jdict = {"men": self.men, "women": self.women,
                 "nights": self.nights, "matchboxes": mbtrans,
                 "bonights": self.bonights}
        if self.cancellednight != -1:
            jdict["cancelled"] = self.cancellednight

        json_object = json.dumps(jdict, indent=4)
        with open(f"data/{self.seasonname}.json", "w") as outfile:
            outfile.write(json_object)


def clean_name(name):
    # remove numbers from footnotes
    cname = re.sub(r"\d", "", name)
    cname = re.sub(r"\[Note \d\]", "", cname)
    # remove second name
    cname = re.sub(r"([a-zA-Z\u00E9]+) ([\"A-Z](?!\.)[\"a-zA-z\u00E9 ]+)",
                   r"\1", cname)

    cname = re.sub(r"([A-Za-z\u00E9]+) & ([A-Za-z\u00E9]+)", r"\1", cname)
    cname = re.sub(r"([A-Za-z\u00E9]+)/([A-Za-z\u00E9]+)", r"\1", cname)
    if cname in pseudonyms_spellingerros:
        cname = pseudonyms_spellingerros[cname]
    return cname


def parse_cast_table(table):
    names = table["Cast member"].to_list()
    names = list(map(clean_name, names))
    return names


if __name__ == "__main__":
    wikiurl = "https://en.wikipedia.org/wiki/Are_You_the_One%3F_(German_TV_series)"

    tables = pd.read_html(wikiurl)
    casts = list(
        filter(lambda ct: "Cast member" in ct.columns.to_list(), tables))
    nights = list(filter(lambda ct: isinstance(ct.columns, pd.MultiIndex)
                         and ('Matching Night', '1') in ct.columns.tolist(),
                         tables))
    boxes = list(
        filter(lambda ct: "Match Box" in ct.columns.to_list() or "MatchBox" in ct.columns.to_list(),
               tables))

    normalo2020 = "normalo2020", casts[0], casts[1], boxes[0], nights[0], nights[1]
    normalo2021 = "normalo2021", casts[2], casts[3], boxes[1], nights[2], nights[3]
    normalo2022 = "normalo2022", casts[4], casts[5], boxes[2], nights[4], nights[5]
    normalo2023 = "normalo2023", casts[6], casts[7], boxes[3], nights[6], nights[7]
    vip2021 = "vip2021", casts[8], casts[9], boxes[4], nights[8], nights[9]
    vip2022 = "vip2022", casts[10], casts[11], boxes[5], nights[10], nights[11]
    vip2023 = "vip2023", casts[12], casts[13], boxes[6], nights[12], nights[13]

    aytoseason = Season_Parser(*vip2022)
    aytoseason.write_json()
