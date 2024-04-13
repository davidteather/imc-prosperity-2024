import itertools
from typing import List

prices = {
    "Pizza": {"Pizza": 1, "Wasabi": 0.48, "Snowball": 1.52, "Shells": 0.71},
    "Wasabi": {"Pizza": 2.05, "Wasabi": 1, "Snowball": 3.26, "Shells": 1.56},
    "Snowball": {"Pizza": 0.64, "Wasabi": 0.3, "Snowball": 1, "Shells": 0.46},
    "Shells": {"Pizza": 1.41, "Wasabi": 0.61, "Snowball": 2.08, "Shells": 1},
}

goods = ["Pizza", "Wasabi", "Snowball", "Shells"]


def calc_money(trades: List[str], amount: float) -> float:
    for i in range(0, len(trades) - 1):
        good_from = trades[i]
        good_to = trades[i + 1]
        amount *= prices[good_from][good_to]
    return amount


if __name__ == '__main__':
    start_amount = 2_000_000
    result = []
    all_combinations = list(itertools.product(goods, repeat=4)) + \
        list(itertools.product(goods, repeat=3)) + \
        list(itertools.product(goods, repeat=2)) + \
        list(itertools.product(goods, repeat=1))

    # Go through all the combinations and append their result
    for combination in all_combinations:
        trade = ["Shells"] + list(combination) + ["Shells"]
        result.append((calc_money(trade, start_amount), trade))

    result.sort(key=lambda x: (-x[0], len(x[1])))
    print("Final amount: {}\nAchieved with the combination of: {}".format(result[0][0], result[0][1]))