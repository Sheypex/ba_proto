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
from data_types import PickleOut
from rich.console import Console
import commons


def main():
    rc = commons.rc
    #
    if False:
        models = []
        for f in os.listdir("models"):
            if re.match(".*\.pickle$", f) and f != "bestModel.pickle":
                l: PickleOut = pickle.load(open(f"models/{f}", "br"))
                name, regr, test_conf, deg, full_conf, unknown_conf, train_conf, bonusPickleInfo = l
                if test_conf >= 0 and deg < 4:
                    models.append((name, regr, test_conf, deg, f"models/{f}"))
        models.sort(key=lambda x: -x[2])
        # print(tabulate([[m[3] if len(m) >= 4 else None, m[0], m[2]] for m in models], headers=["degree", "name", "conf"], missingval='N/A', showindex=True))

        indices = [int(round(len(models) - 1 - x * (len(models) - 1), 0)) for x in [a / 10 for a in range(0, 11)]]
        # print(indices)
        models = [m for i, m in enumerate(models) if i in indices]
        # print(tabulate([[m[3] if len(m) >= 4 else None, m[0], m[2]] for m in models], headers=["degree", "name", "conf"], missingval='N/A', showindex=True))
        [print(x[4]) for x in models]
        for i, x in enumerate(models):
            name, _, conf, deg, _ = x
            name = name.replace(" Regression", "")
            name = name.replace(" Regressor", "")
            # name = name.replace(" CV", "")
            models[i] = (name, f"{float(conf):.4f}", deg)
        # print(
        #     tabulate(
        #         [(x[2], x[0], x[1]) for x in models],
        #         headers=("degree", "name", "conf"),
        #         tablefmt="simple",
        #         disable_numparse=True,
        #     )
        # )
    #
    if True:
        cvmodelDir = "filteredFeatureModels"
        models = {"regression": [], "nn": [], "dt": []}
        for f in os.listdir(cvmodelDir):
            if re.match(".*\.pickle$", f) and f != "bestModel.pickle":
                l: PickleOut = pickle.load(open(f"{cvmodelDir}/{f}", "br"))
                name, regr, test_confidence, deg, full_confidence, unknown_confidence, train_confidence, bonusPickleInfo = l
                if True or (test_confidence >= 0 and deg < 4):
                    if "Neural Network" in name:
                        cat = models["nn"]
                    elif "Decision Tree" in name:
                        cat = models["dt"]
                    else:
                        cat = models["regression"]

                m = re.match(".*_CV-(.*?)_U-(.*?)\.pickle$", f)
                cv, unknown = m.group(1, 2)
                if True:
                    cat.append(
                        (
                            name,
                            commons.jamGeomean([test_confidence, full_confidence, train_confidence]),
                            deg,
                            f"{cvmodelDir}/{f}",
                            cv,
                            unknown,
                            test_confidence,
                            full_confidence,
                            train_confidence,
                            unknown_confidence,
                        )
                    )
                del l
                del name
                del test_confidence
                del deg
                del full_confidence
                del unknown_confidence
                del train_confidence
                del bonusPickleInfo
        # cvmodels.sort(key=lambda x: -x[2])
        for cat, col in models.items():
            df = pds.DataFrame(
                col,
                columns=[
                    "name",
                    "gconf",
                    "degree",
                    "path",
                    "cv",
                    "uk",
                    "test_conf",
                    "full_conf",
                    "train_conf",
                    "uk_conf",
                ],
            )
            # print(tabulate([[m[3] if len(m) >= 4 else None, m[0], m[5], m[2]] for m in cvmodels], headers=["degree", "name", "cv", "conf"], missingval='N/A', showindex=True))
            with pds.option_context(
                "display.max_rows", None, "display.max_columns", None, "display.width", None, "display.precision", 4,
            ):
                df.sort_values(by="path", inplace=True, ignore_index=True)
                # hasAllCVs = df.groupby(["degree", "name"])["cv"].agg(lambda c: len(c) == 5)
                # print(hasAllCVs)
                # print(df)
                # df = df.join(hasAllCVs, on=["degree", "name"], rsuffix="_p")
                # df = df[df["cv_p"]]
                # df.drop("cv_p", axis=1, inplace=True)
                # print(df)
                m = df.groupby(["degree", "name"])["gconf"].aggregate(commons.jamGeomean)
                # m.sort_values(ascending=False, inplace=True)
                # print(m)
                df = df.join(m, on=["degree", "name"], rsuffix="_m")
                df.sort_values(by=["gconf_m"], ascending=False, inplace=True)
                df = df.reset_index(drop=True)
                print(df)
                #
                best = df.head(5).values
                # [rc.print(x[3]) for x in best]
    #
    if False:
        pModels = list()
        with open("percentileModels", "r") as pModelsF:
            for line in pModelsF:
                # print(line)
                with open(line.strip(), "br") as f:
                    name, _, test_conf, deg, full_conf = pickle.load(f)
                    pModels.append((deg, name, f"{test_conf:.4f}"))
        print(tabulate(pModels, headers=("Degree", "Name", "Conf"), tablefmt="latex", disable_numparse=True,))
    #
    if False:
        cvmodels = []
        cvmodelDir = "cvmodels"
        for f in os.listdir(cvmodelDir):
            if re.match(".*\.pickle$", f) and f != "bestModel.pickle":
                with open(f"{cvmodelDir}/{f}", "br") as pF:
                    l: PickleOut = pickle.load(pF)
                (
                    name,
                    _,
                    test_confidence,
                    deg,
                    full_confidence,
                    unknown_confidence,
                    train_confidence,
                    bonusPickleInfo,
                ) = l
                m = re.match(".*_CV-(.*?)_U-(.*?)\.pickle$", f)
                cv, unknown = m.group(1, 2)
                if (test_confidence >= -1 and full_confidence >= -10 and train_confidence >= 0) or False:
                    cvmodels.append(
                        (
                            name,
                            jamGeomean([test_confidence, full_confidence, train_confidence]),
                            deg,
                            f"{cvmodelDir}/{f}",
                            cv,
                            unknown,
                            test_confidence,
                            full_confidence,
                            train_confidence,
                            unknown_confidence,
                        )
                    )
                del l
                del name
                del test_confidence
                del deg
                del full_confidence
                del unknown_confidence
                del train_confidence
                del bonusPickleInfo
        # cvmodels.sort(key=lambda x: -x[2])
        df = pds.DataFrame(
            cvmodels,
            columns=[
                "name",
                "gconf",
                "degree",
                "path",
                "cv",
                "uk",
                "test_conf",
                "full_conf",
                "train_conf",
                "uk_conf",
            ],
        )
        # print(tabulate([[m[3] if len(m) >= 4 else None, m[0], m[5], m[2]] for m in cvmodels], headers=["degree", "name", "cv", "conf"], missingval='N/A', showindex=True))
        with pds.option_context(
            "display.max_rows", None, "display.max_columns", None, "display.width", None, "display.precision", 4,
        ):
            df.sort_values(by="path", inplace=True, ignore_index=True)
            hasAllCVs = df.groupby(["degree", "name"])["cv"].agg(lambda c: len(c) == 20)
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
            # print(df)
            #
            numGroups = len(df) / 20
            indices = [
                int(round(((numGroups - 1) - x * (numGroups - 1)), 0) * 20) for x in [a / 10 for a in range(0, 11, 5)]
            ]
            indices = [x for i in indices for x in [i + o for o in range(0, 20)]]
            # print(indices)
            cvmodels = [m for i, m in enumerate(df.values) if i in indices]
            # CL.print(tabulate([[m[2], m[3], m[1], m[10]] for m in cvmodels], headers=["degree", "name", "gconf", "gconf_m"], missingval='N/A', showindex=True))
            [rc.print(x[3]) for x in cvmodels]


if __name__ == "__main__":
    main()
