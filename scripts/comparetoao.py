import json
from collections import Counter

from ayto import Pair, Solution, Solver, get_solver

# pyright: standard


def comparetoao(solver: Solver, sols: list[Solution], end: int, icl: bool):
    with open("solutions.txt", "r") as f:
        lines = map(eval, f.readlines())

    def parse_sol(ld: dict[str, list[str]]):
        pairs: set[Pair] = set()
        for l, rs in ld.items():
            for r in rs:
                pairs.add((l, r))
        return Solution(set(pairs))

    # solsao = list(map(parse_sol, lines))
    aosols = [Solution(s) for s in lines]

    print("lengths", len(aosols), len(sols))
    count = count2 = 0
    reasons: list[str] = []
    for aosol in aosols:
        if vio := solver.find_violation(aosol, end, icl):
            # print("not possible", vio, aosol)
            for l, r in aosol:
                if nm := solver.no_match_reason(l, r, end, icl):
                    print("no match reason", l, r, nm)
            count += 1
            reasons.append(vio)

    print(
        f"solution in solsao not possible: {count}; solutions in solsao not in sols {count2}"
    )
    print(len(reasons), Counter(reasons))

    count = count2 = 0
    reasons = []
    for s in sols:
        if s not in aosols:
            spd = solver.find_violation(s, end, False)
            count += 1
            # print(s)
            if spd is not None:
                reasons.append(spd)
                # print(spd)
                count2 += 1
    print(Counter(reasons))
    print(f"sols not in solsao: {count}; {count2}")

    for s in aosols:
        if s not in sols:
            print(s)
            # if vio :=solver.find_violation(s, end, True):
            #     print("not possible", vio, s)
            #     for l, r in s:
            #         if nm := solver.no_match_reason(l, r, end, True):
            #             print("no match reason", l, r, nm)
            # print(solver.find_violation(s, end+1, include_night=False))
            break
    # print(Counter(nomatchp))


def check_ao_data(sols: list[Solution], end, sn):

    with open("ayto_data.json", encoding="utf-8") as f:
        all_data = json.load(f)

    season_data = all_data[sn]
    weeks = season_data["weeks"]
    # print("end", end)
    for week in weeks[: end + 1]:
        # print("week number", week["number"])
        for event in week["events"]:
            if event["type"] == "box":
                a, b = event["pair"]
                if event["result"] == "yes" and not (a, b) in sols:
                    return False, event
                if event["result"] == "no" and (a, b) in sols:
                    return False, event
            elif event["type"] == "matching_night":
                lights = sum(1 for a, b in event["pairs"] if (a, b) in sols)
                if lights != event["lights"]:
                    return False, event
            # else:
            #     print("other event", event)
    return True, None


def iterate_ao_data(sn):

    with open("ayto_data.json", encoding="utf-8") as f:
        all_data = json.load(f)

    season_data = all_data[sn]
    weeks = season_data["weeks"]
    # for s in season_data:
    #     if s == "weeks":
    #         continue
    #     print(s, ":", season_data[s])

    for week in weeks:
        # print("week number", week["number"])
        mn = True
        for event in week["events"]:

            if event["type"] == "box":
                a, b = event["pair"]
                # print(event["type"], a, b, mn)
                # print(f"matchbox before: {mn}")
                # if not mn:
                #     print(f"matchbox end of episode {a,b} { week["number"]}")
            elif event["type"] == "matching_night":
                # print(event["type"])
                mn = False
            else:
                print(event)

        # print("-----")


if __name__ == "__main__":
    # uv run python -m scripts.comparetoao
    ao_allseasons = [
        "s1",
        "s1vip",
        "s2",
        "s2vip",
        "s3",
        "s3vip",
        "s4",
        "s4vip",
        "s5",
        "s5vip",
        "s6",
        "s7",
        "s6vip",
    ]
    # for sn in ao_allseasons:
    #     iterate_ao_data(sn)
    sn = "vip2023"
    sols = get_solver(sn).solve(7, False)
    comparetoao(get_solver(sn), sols, 7, False)
    # df = pd.read_csv("analytics/num_solutions.csv", index_col=0, header=0)
    # df.columns = df.columns.astype(int)
    # df.index = df.index.str.strip()

    # i = 7

    # l = df.loc[f"normalo202{i-1}"].to_list()  # pyright: ignore[reportCallIssue]
    # s = ""
    # for x in l:
    #     s += f"0, {x}, "
    # print(s)

    # sn = f"s{i}"
    # df = pd.read_csv(f"analytics/{sn}_event_counts.csv", index_col=0, header=0)
    # print(df["solutions_before"].to_list())  # pyright: ignore[reportCallIssue]
