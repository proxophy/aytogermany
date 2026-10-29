import json
from collections import Counter

from ayto import Pair, Solution, Solver

# pyright: standard

def comparetoao(solver: Solver, sols: list[Solution], end: int):
    with open("aodata.txt", "r") as f:
        lines = map(eval, f.readlines())

    def parse_sol(ld: dict[str, list[str]]):
        pairs: set[Pair] = set()
        for l, rs in ld.items():
            for r in rs:
                pairs.add((l, r))
        return Solution(set(pairs))

    solsao = list(map(parse_sol, lines))

    print("lengths", len(solsao), len(sols))
    count = count2 = 0
    reasons: list[str] = []
    for s in solsao:
        spd = solver.find_violation(s, end, True)
        if spd is not None:
            count += 1
            reasons.append(spd)
        elif s not in sols:
            count2 += 1
            print(s)
            # break

    print(
        f"solution in solsao not possible: {count}; solutions in solsao not in sols {count2}"
    )
    print(len(reasons), Counter(reasons))

    count = count2 = 0
    reasons = []
    for s in sols:
        if s not in solsao:
            spd = solver.find_violation(s, end, False)
            count += 1
            # print(s)
            if spd is not None:
                reasons.append(spd)
                # print(spd)
                count2 += 1
    print(Counter(reasons))
    print(f"sols not in solsao: {count}; {count2}")
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
    print(season_data.keys())

    for week in weeks: 
        print("week number", week["number"])
        for event in week["events"]:
        
            if event["type"] == "box":
                a, b = event["pair"]
                print(event["type"], a,b)
            else:
                print(event["type"])
                
            # elif event["type"] == "matching_night":
            #     lights = sum(1 for a, b in event["pairs"] if (a, b) in sols)
            #     if lights != event["lights"]:
            #         return False, event
            # else:
            #     print("other event", event)
        print("-----")
    return 


if __name__ == "__main__":
    # uv run python -m scripts.comparetoao
    iterate_ao_data("s5vip")