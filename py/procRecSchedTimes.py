import cProfile
import os
import pickle
import pstats
import re
from collections import namedtuple
from typing import List
from multiprocessing import Pool
import networkx as NX
import pandas as pds
import matplotlib.pyplot as plt
import itertools
import math
import re
import statistics
from pprint import pprint

import scipy
from scipy import stats as scistats
from sklearn import linear_model, preprocessing
from tabulate import tabulate
from pathlib import Path
import pickle
import random
import argparse
import sys
from db_actions import db_actions
import numpy as np
from my_yaml import yaml_load, yaml_dump
from alive_progress import alive_bar
import seaborn as sns

TR = namedtuple('TableRow', ["cluster", "wfName", "methodName", "res"])
BigTR = namedtuple("BigTableRow", ["regModel", "cluster", "clusterSize", "wfName", "methodName", "res"])


def list_compare(a, b):
    if type(a) != type(b):
        return False
    if type(a) != list:
        return a == b
    if len(a) != len(b):
        return False
    for a_, b_ in zip(a, b):
        if not list_compare(a_, b_):
            return False
    return True


def load_from_file(f):
    modelDegLUT = {
        'linModel' : 1,
        'quadModel': 2,
        'cubeModel': 3,
        'tetModel' : 4,
        'penModel' : 5
    }
    shorterMethodNamesLUT = {
        'randomScheduler'     : 'random',
        'recommenderScheduler': 'recommender'
    }
    shortToLongNameLUT = {'Linear'       : 'Ordinary Least Squares',
                          'SVR-linear'   : 'SVR - linear',
                          'SVR-poly'     : 'SVR - poly',
                          'SVR-rbf'      : 'SVR - rbf',
                          'SVR-sigmoid'  : 'SVR - sigmoid',
                          'Lasso'        : 'Lasso',
                          'LassoCV'      : 'Lasso',
                          'Ridge'        : 'Ridge',
                          'RidgeCV'      : 'Ridge',
                          'ElasticNet'   : 'Elastic Net',
                          'ElasticNetCV' : 'Elastic Net',
                          'BayesianRidge': 'Bayesian Ridge',
                          'ARD'          : 'Automatic Relevance Determination',
                          'SGD'          : 'Stochastic Gradient Descent',
                          'PA'           : 'Passive Aggressive',
                          'Huber'        : 'Huber',
                          'TheilSen'     : 'Theil Sen'}
    #
    pFile = Path(f)
    with open(pFile, "br") as tmp:
        pl = pickle.load(tmp)
    shortName = re.match("(.*/)*(.*?)\.recSchedTimes\.pickle", f).group(2)
    modelDeg = modelDegLUT[shortName.split(".")[0]]
    shortName = f"{modelDeg}/{shortToLongNameLUT[shortName.split('.')[1]]}"
    for pre, deg in modelDegLUT.items():
        if pre in shortName:
            shortName = shortName.replace(pre + ".", f"{deg}/")
            break
    for clusterName, cluster in pl.items():
        for wfName, wf in cluster.items():
            for methodName, res in wf.items():
                for longN, rep in shorterMethodNamesLUT.items():
                    if longN in methodName:
                        mName = methodName.replace(longN, rep)
                        break
                if type(res) != list:
                    yield BigTR(shortName, clusterName, len(clusterName), wfName, mName, float(res))
                else:
                    for i, r in enumerate(res):  # this only applies to the results of the randomScheduler(V1)
                        yield BigTR("N/A", clusterName, len(clusterName), wfName, mName + f"P{i}", float(r))
                    yield BigTR("N/A", clusterName, len(clusterName), wfName, mName + "Avg", float(statistics.mean(res)))


def load_from_dir(d):
    with alive_bar(len(os.listdir(d)), f"Loading recommenderScheduler results from {d}") as bar:
        for f in os.listdir(d):
            yield from load_from_file(f"{d}/{f}")
            bar()


def main():
    # Flags:
    cvmode = False
    #
    saveLoc = "recSchedTimes"
    btFile = Path("./recSchedBigTable.pickle")
    if not btFile.is_file():
        bigtable = pds.DataFrame(load_from_dir(saveLoc), columns=["regModel", "cluster", "clusterSize", "wfName", "method", "res"])
        with open(btFile, "bw") as f:
            pickle.dump(bigtable, f)
    else:
        with open(btFile, "br") as f:
            bigtable = pickle.load(f)
    bigtable = bigtable.sample(frac=5 / 100, random_state=0, axis=0)
    wfNames = bigtable.wfName.unique()
    # print(bigtable)
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    #
    with pds.option_context('display.max_rows', None,
                            'display.max_columns', None,
                            'display.width', None,
                            'display.precision', 4, ):
        if True:
            g = bigtable.groupby(["method"])
            desc = g["res"].describe(  # percentiles=[x / 10 for x in range(1, 10)],
                include="all")
            desc.sort_values(by=["mean"], axis=0, inplace=True, ascending=True)
            pprint(desc)
            g = sns.catplot(x="res", y="method", orient="h",
                            order=desc.index,
                            # hue="wfName",  # palette=["m", "g"],
                            # row="wfName",
                            # col="method",
                            data=bigtable,
                            kind="violin",
                            aspect=1.5
                            # height=20
                            )
            g.set(xlabel=None, ylabel=None)
            plt.savefig("./fig/res_Vs_method_Violin.plot.pdf")
            plt.savefig("./fig/res_Vs_method_Violin.plot.png")
            plt.show()
            #
            g = bigtable.groupby(["regModel"])
            desc = g["res"].describe(  # percentiles=[x / 10 for x in range(1, 10)],
                include="all")
            desc.sort_values(by=["mean"], axis=0, inplace=True, ascending=True)
            pprint(desc)
            g = sns.catplot(x="res", y="regModel", orient="h",
                            order=desc.index,
                            # hue="wfName",  # palette=["m", "g"],
                            # row="wfName",
                            # col="method",
                            data=bigtable,
                            kind="violin",
                            aspect=1.5
                            # height=20
                            )
            g.set(xlabel=None, ylabel=None)
            plt.savefig("./fig/res_Vs_regModel_Violin.plot.pdf")
            plt.savefig("./fig/res_Vs_regModel_Violin.plot.png")
            plt.show()
            #
            exit(0)
        #
        g = bigtable.groupby(["regModel", "method"])
        desc = g["res"].describe(  # percentiles=[x / 10 for x in range(1, 10)],
            include="all")
        desc.sort_values(by=["mean"], axis=0, inplace=True, ascending=True)
        pprint(desc)
        #
        # g = bigtable.groupby(["regModel", "clusterSize", "method"])
        # desc = g["res"].describe(percentiles=[x / 10 for x in range(1, 10)], include="all")
        # desc.sort_values(by=["mean"], axis=0, inplace=True, ascending=True)
        # pprint(desc)
        #
        for wf in wfNames:
            print(f"""
            {"#" * (len(wf) + 4)}
            # {wf} #
            {"#" * (len(wf) + 4)}""")
            f = bigtable.query("wfName == @wf")
            #
            g = f.groupby(["method"])
            desc = g["res"].describe(  # percentiles=[x / 10 for x in range(1, 10)],
                include="all")
            desc.sort_values(by=["mean"], axis=0, inplace=True, ascending=True)
            pprint(desc)
            #
            g = f.groupby(["regModel"])
            desc = g["res"].describe(  # percentiles=[x / 10 for x in range(1, 10)],
                include="all")
            desc.sort_values(by=["mean"], axis=0, inplace=True, ascending=True)
            pprint(desc)
            #
            g = f.groupby(["regModel", "method"])
            desc = g["res"].describe(  # percentiles=[x / 10 for x in range(1, 10)],
                include="all")
            desc.sort_values(by=["mean"], axis=0, inplace=True, ascending=True)
            pprint(desc)


if __name__ == '__main__':
    main()
