import math
import os
import pickle
import re
import pandas as pds
import sys
from pprint import pprint
from scipy import stats as scistats
from tabulate import tabulate
import statistics


# jamGeomean <- function
# (x,
#  na.rm=TRUE,
#  ...)
# {
#    ## Purpose is to calculate geometric mean while allowing for
#    ## positive and negative values
#    x2 <- mean(log2(1+abs(x))*sign(x));
#    sign(x2)*(2^abs(x2)-1);
# }
# taken from: https://jmw86069.github.io/splicejam/reference/jamGeomean.html
def jamGeomean(iterable):
    assert len(iterable) > 0
    step1 = [math.log(1 + abs(x), 2) * math.copysign(1, x) for x in iterable]
    m = statistics.mean(step1)
    return math.copysign(1, m) * ((2 ** abs(m)) - 1)


def main():
    if False:
        models = []
        for f in os.listdir("models"):
            if re.match(".*\.pickle$", f) and f != "bestModel.pickle":
                l = pickle.load(open(f"models/{f}", "br"))
                name, regr, test_conf, deg, full_conf = l
                if test_conf >= 0:
                    models.append((name, regr, test_conf, deg, f"models/{f}"))
        models.sort(key=lambda x: -x[2])
        # print(tabulate([[m[3] if len(m) >= 4 else None, m[0], m[2]] for m in models], headers=["degree", "name", "conf"], missingval='N/A', showindex=True))

        indices = [int(round(len(models) - 1 - x * (len(models) - 1), 0)) for x in [a / 10 for a in range(0, 11)]]
        # print(indices)
        models = [m for i, m in enumerate(models) if i in indices]
        # print(tabulate([[m[3] if len(m) >= 4 else None, m[0], m[2]] for m in models], headers=["degree", "name", "conf"], missingval='N/A', showindex=True))
        [print(x[4]) for x in models]
    #
    cvmodels = []
    for f in os.listdir("cvmodels"):
        if re.match(".*\.pickle$", f) and f != "bestModel.pickle":
            l = pickle.load(open(f"cvmodels/{f}", "br"))
            name, regr, test_conf, deg, full_conf = l
            cv = re.match(".*_(.*?)\.pickle", f).groups(1)
            if test_conf >= -1 and full_conf >= -1:
                cvmodels.append((name, jamGeomean([test_conf, full_conf]), deg, f"cvmodels/{f}", cv, test_conf, full_conf))
    # cvmodels.sort(key=lambda x: -x[2])
    df = pds.DataFrame(cvmodels, columns=["name", "gconf", "degree", "path", "cv", "test_conf", "full_conf"])
    # print(tabulate([[m[3] if len(m) >= 4 else None, m[0], m[5], m[2]] for m in cvmodels], headers=["degree", "name", "cv", "conf"], missingval='N/A', showindex=True))
    with pds.option_context('display.max_rows', None,
                            'display.max_columns', None,
                            'display.width', None,
                            'display.precision', 4, ):
        df.sort_values(by="path", inplace=True, ignore_index=True)
        hasAllCVs = df.groupby(["degree", "name"])["cv"].agg(lambda c: len(c) == 5)
        # print(hasAllCVs)
        # print(df)
        df = df.join(hasAllCVs, on=["degree", "name"], rsuffix="_p")
        df = df[df["cv_p"]]
        df.drop("cv_p", axis=1, inplace=True)
        # print(df)
        m = df.groupby(["degree", "name"])["gconf"].aggregate(jamGeomean)
        # m.sort_values(ascending=False, inplace=True)
        # print(m)
        df = df.join(m, on=["degree", "name"], rsuffix="_m")
        df.sort_values(by=["gconf_m"], ascending=False, inplace=True)
        df = df.reset_index(drop=True)
        print(df)
        #
        numGroups = len(df) / 5
        indices = [int(round(((numGroups - 1) - x * (numGroups - 1)), 0) * 5) for x in [a / 10 for a in range(0, 11, 5)]]
        indices = [x for i in indices for x in [i, i + 1, i + 2, i + 3, i + 4]]
        # print(indices)
        cvmodels = [m for i, m in enumerate(df.values) if i in indices]
        # print(tabulate([[m[3] if len(m) >= 4 else None, m[4], m[2]] for m in cvmodels], headers=["degree", "name", "gconf"], missingval='N/A', showindex=True))
        [print(x[3]) for x in cvmodels]


if __name__ == '__main__':
    main()
