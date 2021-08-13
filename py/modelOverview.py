import math
import os
import pickle
import re
import sys
from pprint import pprint

from tabulate import tabulate


def main():
    models = []
    for f in os.listdir("models"):
        if re.match(".*\.pickle$", f) and f != "bestModel.pickle":
            l = pickle.load(open(f"models/{f}", "br"))
            name, regr, conf, deg = l
            if conf >= 0:
                models.append((name, regr, conf, deg, f"models/{f}"))
    models.sort(key=lambda x: -x[2])
    # print(tabulate([[m[3] if len(m) >= 4 else None, m[0], m[2]] for m in models], headers=["degree", "name", "conf"], missingval='N/A', showindex=True))

    indices = [int(round(len(models) - 1 - x * (len(models) - 1), 0)) for x in [a / 10 for a in range(0, 11)]]
    # print(indices)
    models = [m for i, m in enumerate(models) if i in indices]
    # print(tabulate([[m[3] if len(m) >= 4 else None, m[0], m[2]] for m in models], headers=["degree", "name", "conf"], missingval='N/A', showindex=True))
    [print(x[4]) for x in models]


if __name__ == '__main__':
    main()
