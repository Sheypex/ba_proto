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


def main():
    saveLoc = "recSchedTimes"
    data = dict()
    regModelNames = list()
    bigtable = list()
    with alive_bar(len(os.listdir(saveLoc)), f"Loading recommenderScheduler times from {saveLoc}") as bar:
        for f in os.listdir(saveLoc):
            pFile = Path(f"{saveLoc}/{f}")
            pl = pickle.load(open(pFile, "br"))
            # table = list()
            btrs = list()
            shortName = re.match("(.*?)\.recSchedTimes\.pickle", f).group(1)
            bar.text(shortName)
            for clusterName, cluster in pl.items():
                for wfName, wf in cluster.items():
                    for methodName, res in wf.items():
                        if type(res) != list:
                            # table.append(TR(clusterName, wfName, methodName, float(res)))
                            btrs.append(BigTR(shortName, clusterName, len(clusterName), wfName, methodName, float(res)))
                        else:
                            # table.append(TR(clusterName, wfName, methodName, [float(r) for r in res]))
                            btrs += [BigTR("", clusterName, len(clusterName), wfName, methodName + f"P{i}", float(r)) for i, r in
                                     enumerate(res)]  # this only applies to the results of the randomScheduler(V1)
                            # print(tabulate(table))
            # df = pds.DataFrame(table, columns=["cluster", "wfName", "method", "res"])
            # data[shortName] = df
            regModelNames.append(shortName)
            bigtable += btrs
            bar()
    bigtable = pds.DataFrame(bigtable, columns=["regModel", "cluster", "clusterSize", "wfName", "method", "res"])
    # print(bigtable)
    with pds.option_context('display.max_rows', None,
                            'display.max_columns', None,
                            'display.width', None,
                            'display.precision', 4, ):
        if False:
            g = bigtable.groupby(["method"])
            desc = g["res"].describe(percentiles=[x / 10 for x in range(1, 10)], include="all")
            pprint(desc)
            #
            g = bigtable.groupby(["regModel"])
            desc = g["res"].describe(percentiles=[x / 10 for x in range(1, 10)], include="all")
            pprint(desc)
        #
        g = bigtable.groupby(["regModel", "method"])
        desc = g["res"].describe(percentiles=[x / 10 for x in range(1, 10)], include="all")
        desc.sort_values(by=["mean"], axis=0, inplace=True, ascending=True)
        pprint(desc)
        #
        # g = bigtable.groupby(["regModel", "clusterSize", "method"])
        # desc = g["res"].describe(percentiles=[x / 10 for x in range(1, 10)], include="all")
        # desc.sort_values(by=["mean"], axis=0, inplace=True, ascending=True)
        # pprint(desc)
        #
        g = bigtable.groupby(["regModel", "wfName", "method"])
        desc = g["res"].describe(percentiles=[x / 10 for x in range(1, 10)], include="all")
        desc.sort_values(by=["mean"], axis=0, inplace=True, ascending=True)
        pprint(desc)


if __name__ == '__main__':
    main()
