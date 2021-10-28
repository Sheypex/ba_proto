import cProfile
import pickle
import pstats
import re
from abc import ABC, abstractmethod
from builtins import property
from collections import namedtuple
from functools import cache
from typing import List, Union, Optional
from multiprocessing import Pool
import networkx as NX
import pandas as pds
import matplotlib.pyplot as plt
import itertools
import math
import re
import statistics
from pprint import pprint

import rich.progress
from sklearn import linear_model, preprocessing
from tabulate import tabulate
from pathlib import Path
import pickle
from data_types import PickleOut
import random
import argparse
import sys
from db_actions import db_actions
import numpy as np
from my_yaml import yaml_load, yaml_dump
from alive_progress import alive_bar
from rich.console import Console
import atexit
from rich.traceback import install as niceTracebacks
from rich.progress import Progress as rProgress
import commons
import rich.pretty
import rich.panel
import rich.table
import rich.text

rc = commons.rc


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--csv", type=str, action="store", dest="csvPath")
    argp.add_argument("regModelPath", action="store", type=str)
    argp.add_argument("--saveLoc", type=str, action="store", dest="saveLoc", default="\0")
    cliArgs = argp.parse_args(sys.argv[1:])
    if cliArgs.saveLoc == "\0":
        cliArgs.saveLoc = "exports"
    pFile = Path(cliArgs.saveLoc)
    if not pFile.exists():
        pFile.mkdir()
    else:
        assert pFile.is_dir()
    #
    if re.match(".*\.pickle", cliArgs.regModelPath):
        regModels = [cliArgs.regModelPath]
    else:
        with open(cliArgs.regModelPath, "r") as f:
            regModels = [l.strip() for l in f.readlines()]
    #
    if cliArgs.csvPath:
        predBase = None
        try:  # TODO i believe an error while opening prevents the context manager from closing the file, leading to "f" being a bricked variable..?
            with open(cliArgs.csvPath, "r") as f:
                predBase = pds.read_csv(f)  # already filtered for realtime > 1000)
        except:
            pass
        try:
            with open(cliArgs.csvPath + ".csv", "r") as f:
                predBase = pds.read_csv(f)
        except:
            pass
        if predBase is None:
            raise Exception(f"Couldn't open csv file {cliArgs.csvPath!r} or {(cliArgs.csvPath + '.csv')!r}")
    else:
        with db_actions.connect() as conn:
            predBase = pds.read_sql('SELECT * FROM "averageRuntimesPredictionBase1000"', conn)
    #
    with open("nodeConfigIdLookup.yaml", "r") as f:
        t = yaml_load(f)
        nodeIDLUT = {v: k for k, v in t.items()}
    predBase["nodeName"] = predBase["nodeConfig"].transform(lambda x: nodeIDLUT[x])
    #
    wfNames = list(predBase["wfName"].unique())
    # valid instances: [165, 193] \ {174,177}  # c5.- (174) and c5a.large (177) are out
    allInstances = list(range(165, 194))
    allInstances.remove(174)
    allInstances.remove(177)
    #
    for regModelFile in regModels:
        try:
            with open(regModelFile, "br") as f:
                loaded: PickleOut = pickle.load(f)
            (
                name,
                regModel,
                test_confidence,
                polyDeg,
                full_confidence,
                unknown_confidence,
                train_confidence,
                bonusPickleInfo,
            ) = loaded
        except Exception as e:
            rc.log(f"Could not load regModel from pickle at {regModelFile!r} with error: {e}")
            rc.print_exception()
            exit(1)
        #
        scale = preprocessing.StandardScaler().fit(
            predBase.drop(["wfName", "taskName", "nodeConfig", "realtime", "rank", "nodeName"], axis=1)
        )
        poly = preprocessing.PolynomialFeatures(degree=polyDeg, interaction_only=True, include_bias=polyDeg > 1).fit(
            predBase.drop(["wfName", "taskName", "nodeConfig", "realtime", "rank", "nodeName"], axis=1)
        )
        scale2 = preprocessing.StandardScaler().fit(
            poly.transform(
                scale.transform(
                    predBase.drop(["wfName", "taskName", "nodeConfig", "realtime", "rank", "nodeName"], axis=1)
                )
            )
        )
        #
        for wfName in wfNames:
            predBaseFilt = predBase.query("(wfName == @wfName)")
            predBaseFilt = predBaseFilt.reset_index(drop=True)
            predRankRes = regModel.predict(
                scale2.transform(
                    poly.transform(
                        scale.transform(
                            predBaseFilt.drop(
                                ["wfName", "taskName", "nodeConfig", "realtime", "rank", "nodeName"], axis=1,
                            )
                        )
                    )
                )
            )
            predRankRes = pds.DataFrame(predRankRes, columns=["rank"])
            predRankRes = pds.concat([predBaseFilt[["wfName", "taskName", "nodeName"]], predRankRes], axis=1)
            #
            predRealtime = predBaseFilt[["wfName", "nodeName", "taskName", "realtime"]]
            #
            wfShortName = re.compile("nfcore/(\w+):.*").match(wfName).group(1)
            # CL.print(f"{'-=' * 5}{wfShortName:^{len(wfShortName) + 4}}{'=-' * 5}")
            predRankRes.to_csv(f"{cliArgs.saveLoc}/ranks_{wfShortName}_{name}_{polyDeg}.csv", index=False)
            predRealtime.to_csv(f"{cliArgs.saveLoc}/realtime_{wfShortName}_{name}_{polyDeg}.csv", index=False)


if __name__ == "__main__":
    main()
