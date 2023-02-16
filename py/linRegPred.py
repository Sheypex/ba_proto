import functools
import itertools
import math
import re
import statistics
from functools import reduce
from pprint import pprint
from typing import (
    List,
    Optional,
    Generator,
    NamedTuple,
    Tuple,
    Set,
    Any,
    Dict,
    Union,
    Callable,
    Iterable,
    Sequence,
)

import rich.panel
import sklearn.base
from alive_progress import alive_bar
from numpy.distutils.misc_util import is_sequence
from tqdm import tqdm, trange
from log_symbols import LogSymbols

import overrides as overrides
import pandas as pds
import tabulate as tabulate
from collections import namedtuple
from sklearn import linear_model, metrics, preprocessing, model_selection, svm
from tabulate import tabulate
from pathlib import Path
import pickle
from halo import Halo
import random
import argparse
import sys
from sklearn.experimental import enable_halving_search_cv
from sklearn import neural_network, tree
from sklearn.model_selection import (
    HalvingGridSearchCV,
    RandomizedSearchCV,
    HalvingRandomSearchCV,
)
import scipy.stats as scistats

import rich.progress
import commons
from data_types import PickleOut
from rich.traceback import install as niceTracebacks

from db_actions import db_actions
import numpy as np


########
# TrainTestUnknownSplit = namedtuple("TrainTestUnknownSplit", ["X_train", "y_train", "X_test", "y_test", "X_unknown", "y_unknown", "X_trans", "ukwfs", "cvwfs"])


class TrainTestUnknownSplit(NamedTuple):
    X_train: List[List[float]]
    y_train: List[List[float]]
    X_test: List[List[float]]
    y_test: List[List[float]]
    X_unknown: List[List[float]]
    y_unknown: List[List[float]]
    X_trans: Tuple[preprocessing.StandardScaler, preprocessing.PolynomialFeatures, preprocessing.StandardScaler]
    ukwfs: List[str]
    cvwfs: List[str]


########


def printStat(txt: str, stat: str = LogSymbols.INFO.value, indent: int = 0) -> None:
    if indent > 0:
        ind = ["  "] * indent
        ind = "".join(ind)
    else:
        ind = ""
    print(f"{ind}{stat} {txt}")


def printSucc(txt: str, indent: int = 0) -> None:
    printStat(txt, LogSymbols.SUCCESS.value, indent)


def printInfo(txt: str, indent: int = 0) -> None:
    printStat(txt, LogSymbols.INFO.value, indent)


def printError(txt: str, indent: int = 0) -> None:
    printStat(txt, LogSymbols.ERROR.value, indent)


def printWarn(txt: str, indent: int = 0) -> None:
    printStat(txt, LogSymbols.WARNING.value, indent)


def printBox(message: str, borderTopBot: str = "#", borderSides: str = "#") -> None:
    print(f"{borderTopBot * (len(message) + 4)}")
    print(f"{borderSides} " + message + f" {borderSides}")
    print(f"{borderTopBot * (len(message) + 4)}")


def getTransformers(
        X: pds.DataFrame, polyDeg: int
) -> Tuple[
    preprocessing.StandardScaler, preprocessing.PolynomialFeatures, preprocessing.StandardScaler,
]:
    scale: preprocessing.StandardScaler = preprocessing.StandardScaler().fit(X)
    if polyDeg > 1:
        poly: preprocessing.PolynomialFeatures = preprocessing.PolynomialFeatures(
                degree=polyDeg, interaction_only=True
        ).fit(X)
    else:
        poly: preprocessing.PolynomialFeatures = preprocessing.PolynomialFeatures(
                degree=polyDeg, interaction_only=True, include_bias=False
        ).fit(X)
    scale2: preprocessing.StandardScaler = preprocessing.StandardScaler().fit(poly.transform(scale.transform(X)))
    return scale, poly, scale2


def applyTransformers(
        X: pds.DataFrame,
        scale: preprocessing.StandardScaler,
        poly: preprocessing.PolynomialFeatures,
        scale2: preprocessing.StandardScaler,
) -> List[List[float]]:
    return scale2.transform(poly.transform(scale.transform(X)))


def getNumSplits(dF: pds.DataFrame, cvSize: int = 0, unknownSize: int = 0, wfs: Optional[List[str]] = None, ) -> int:
    if wfs is None:
        wfs = dF.wfName.unique()
    nSplits = 0
    #
    unknownCombs = [
            (list(ukwfs), [wf for wf in wfs if wf not in ukwfs]) for ukwfs in itertools.combinations(wfs, unknownSize)
    ]
    for ukSplit in unknownCombs:
        ukwfs, kwfs = ukSplit
        #
        cvCombs = [
                (list(cvwfs), [wf for wf in kwfs if wf not in cvwfs]) for cvwfs in itertools.combinations(kwfs, cvSize)
        ]
        for cvSplit in cvCombs:
            nSplits += 1
    return nSplits


def getSplits(
        dF: pds.DataFrame,
        polyDeg: int,
        x_cols: List[str],
        y_cols: List[str],
        cvSize: int = 0,
        unknownSize: int = 0,
        wfs: Optional[List[str]] = None,
        randomOrder: bool = True,
) -> Iterable[TrainTestUnknownSplit]:
    assert 1 <= polyDeg <= 5
    assert 0 <= cvSize <= 4
    assert 0 <= unknownSize <= 4
    assert 0 <= cvSize + unknownSize <= 4
    #
    if wfs is None:
        wfs: List[str] = dF.wfName.unique()
    #
    splits = list()
    #
    unknownCombs: List[Tuple[List[str], List[str]]] = [
            (list(ukwfs), [wf for wf in wfs if wf not in ukwfs]) for ukwfs in itertools.combinations(wfs, unknownSize)
    ]
    for ukSplit in unknownCombs:
        ukwfs, kwfs = ukSplit
        #
        cvCombs: List[Tuple[List[str], List[str]]] = [
                (list(cvwfs), [wf for wf in kwfs if wf not in cvwfs]) for cvwfs in itertools.combinations(kwfs, cvSize)
        ]
        for cvSplit in cvCombs:
            cvwfs, trainwfs = cvSplit
            splits.append((ukwfs, kwfs, cvwfs, trainwfs))
    #
    if randomOrder:
        random.shuffle(splits)
    #
    for split in splits:
        ukwfs, kwfs, cvwfs, trainwfs = split
        #
        rows_known: pds.DataFrame = dF.query("wfName in @kwfs")
        rows_unknown: pds.DataFrame = dF.query("wfName in @ukwfs")
        X_trans = getTransformers(rows_known[x_cols], polyDeg)
        X_scale, X_poly, X_scale2 = X_trans
        #
        X_train = applyTransformers(rows_known.query("wfName in @trainwfs")[x_cols], X_scale, X_poly, X_scale2)
        X_test = (
                X_train
                if len(cvwfs) == 0
                else applyTransformers(rows_known.query("wfName in @cvwfs")[x_cols], X_scale, X_poly, X_scale2)
        )
        X_unknown = list() if len(ukwfs) == 0 else applyTransformers(rows_unknown[x_cols], X_scale, X_poly, X_scale2)
        #
        y_train = rows_known.query("wfName in @trainwfs")[y_cols]
        y_test = y_train if len(cvwfs) == 0 else rows_known.query("wfName in @cvwfs")[y_cols]
        y_unknown = list() if len(ukwfs) == 0 else rows_unknown[y_cols]
        #
        yield TrainTestUnknownSplit(
                X_train, y_train, X_test, y_test, X_unknown, y_unknown, X_trans, ukwfs, cvwfs,
        )


rc = commons.rc
MAXTRIES = 1e4


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("polyDeg", type=int, choices=[1, 2, 3, 4, 5], action="store", default=1)
    argp.add_argument("--csv", type=str, action="store", dest="csvPath")
    argp.add_argument("--recompute", action="store_true", dest="recompute", default=False)
    argp.add_argument("--improve", action="store_true", dest="improve", default=False)
    argp.add_argument("--maxiter", type=int, action="store", dest="maxiter", default=-1)
    argp.add_argument(
            "models",
            action="store",
            nargs="*",
            default="all",
            choices=["all"] + get_model_names() + [f"_{x}" for x in get_model_names()],
    )
    argp.add_argument("--repeat", action="store", type=int, dest="numRepeats", default=1)
    argp.add_argument("--latex", action="store_true", dest="latex", default=False)
    argp.add_argument("--showDone", action="store_true", dest="showDone", default=False)
    argp.add_argument("--models", action="store", dest="modelsPath", default="models")
    argp.add_argument(
            "--WFCV", action="store", type=int, choices=[0, 1, 2, 3, 4], default=0, dest="cvSize",
    )
    argp.add_argument(
            "--WFU", action="store", type=int, choices=[0, 1, 2, 3, 4], default=0, dest="unknownSize",
    )
    argp.add_argument("--sanity-check", action="store_true", dest="sanityCheck", default=False)
    argp.add_argument("--saveBest", action="store_true", dest="saveBest", default=False)
    cliArgs = argp.parse_args(sys.argv[1:])
    doDegree = cliArgs.polyDeg
    doPoly = doDegree > 1
    assert not (
            cliArgs.recompute and cliArgs.improve
    ), "--recompute and --improve are mutually exclusive"  # mutually exclusive
    if (
            type(cliArgs.models) is str
    ):  # in case only one model or 'all' is given make sure its still in the format of a list
        cliArgs.models = [cliArgs.models]
    modelsToProduce: set[str] = set(cliArgs.models)  # parse 'all'
    if "all" in modelsToProduce:
        modelsToProduce.remove("all")
        modelsToProduce = modelsToProduce.union(set(get_model_names()))
    modelsToExclude = []
    for model in modelsToProduce:  # parse excluded models
        if model[0] == "_":  # if model is excluded
            modelsToExclude.append(model[1:])
    for model in modelsToExclude:
        try:
            modelsToProduce.remove(f"_{model}")
            modelsToProduce.remove(model)
        except:
            printWarn(f"Failed to exclude {model}")
    cliArgs.models = list(modelsToProduce)
    cliArgs.modelsPath = f"./{cliArgs.modelsPath}"
    p = Path(cliArgs.modelsPath)
    p.mkdir(exist_ok=True)
    if p.exists() and not p.is_dir():
        raise FileExistsError(
                f"The path {p} exists but is not a directory. It is an invalid location for storing models."
        )
    assert (
            cliArgs.cvSize + cliArgs.unknownSize <= 4
    ), f"The sum of workflows used in the cv folds and tested as unknowns may not exceed 4: {cliArgs.cvSize=} and {cliArgs.unknownSize=} were supplied."
    # print cli config
    # printInfo(f"Running with command line arguments: {cliArgs.__dict__}")
    #

    if cliArgs.csvPath:
        dF = None
        try:
            dF = pds.read_csv(open(cliArgs.csvPath, "r"))  # already filtered for realtime > 1000)
        except:
            pass
        try:
            dF = pds.read_csv(open(cliArgs.csvPath + ".csv", "r"))
        except:
            pass
        if dF is None:
            raise Exception(f"Couldn't open csv file {cliArgs.csvPath!r} or {(cliArgs.csvPath + '.csv')!r}")
    else:
        with db_actions.connect() as conn:
            dF = pds.read_sql('SELECT * FROM "averageRuntimesPredictionBase1000"', conn)
    # print(dF)
    all_x_cols = [
            "build-linux-kernel1",
            "fio2",
            "fio3",
            "fio4",
            "fio5",
            "fio6",
            "fio7",
            "fio8",
            "fio9",
            "iperf10",
            "iperf11",
            "iperf12",
            "iperf13",
            "john-the-ripper14",
            "john-the-ripper15",
            "ramspeed16",
            "ramspeed17",
            "ramspeed18",
            "ramspeed19",
            "ramspeed20",
            "ramspeed21",
            "ramspeed22",
            "ramspeed23",
            "ramspeed24",
            "ramspeed25",
            "stream26",
            "stream27",
            "stream28",
            "stream29",
            "pCpu",
            "cpus",
            "rss",
            "vmem",
            "rchar",
            "wchar",
            "syscr",
            "syscw",
    ]
    filtered_x_cols = ['build-linux-kernel1', 'fio3', 'fio5', 'fio6', 'fio7', 'fio8', 'fio9', 'iperf10', 'iperf11', 'john-the-ripper14', 'john-the-ripper15', 'ramspeed16',
                       'ramspeed17', 'ramspeed18', 'ramspeed19', 'ramspeed20', 'ramspeed21', 'ramspeed22', 'ramspeed23', 'ramspeed24', 'ramspeed25', 'stream26', 'stream27',
                       'stream28', 'stream29']
    x_cols = filtered_x_cols
    y_cols = "rank"
    #
    wfs = dF.wfName.unique()
    wfShortnamesLUT = {wf: re.compile("nfcore/(\w+):.*").match(wf).group(1) for wf in wfs}
    wfLongnamesLUT = {short: long for long, short in wfShortnamesLUT.items()}
    picklePrefixes = {1: "lin", 2: "quad", 3: "cube", 4: "tet", 5: "pen"}
    #
    X = dF[x_cols]
    y = dF[y_cols]
    full_scale, full_poly, full_scale2 = getTransformers(X, doDegree)
    X_full = applyTransformers(X, full_scale, full_poly, full_scale2)
    y_full = y
    #
    numSplits = getNumSplits(dF, cliArgs.cvSize, cliArgs.unknownSize, wfs)
    with commons.stdProgress(rc) as prog:
        totalProg = prog.add_task(
                "Total", total=cliArgs.numRepeats * numSplits * len(get_models(restrict=cliArgs.models)),
        )
        repProg = prog.add_task("Repeats", total=cliArgs.numRepeats)
        splitProg = prog.add_task("Splits", total=numSplits)
        modelProg = prog.add_task("Models", total=len(get_models(restrict=cliArgs.models)))
        #
        for iReps in range(cliArgs.numRepeats):
            rc.rule(f"Repeat {iReps + 1}")
            splits = getSplits(dF, doDegree, x_cols, y_cols, cliArgs.cvSize, cliArgs.unknownSize, wfs)
            splitsDone = 0
            for split in splits:
                (X_train, y_train, X_test, y_test, X_unknown, y_unknown, X_trans, ukwfs, cvwfs,) = split
                assert len(X_train) == len(y_train)
                assert len(X_test) == len(y_test)
                assert len(X_unknown) == len(y_unknown)
                rc.print(
                        rich.panel.Panel(
                                f"{ukwfs=}, {cvwfs=}: {len(X_train)=}, {len(X_test)=}, {len(X_unknown)=}",
                                title="Split",
                                title_align="left",
                        )
                )
                if len(cvwfs) > 0:
                    cvsName = "+".join([wfShortnamesLUT[cv] for cv in cvwfs])
                else:
                    cvsName = "None"
                if len(ukwfs) > 0:
                    uksName = "+".join([wfShortnamesLUT[cv] for cv in ukwfs])
                else:
                    uksName = "None"
                doModels = fit_models(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        X_unknown,
                        y_unknown,
                        X_full,
                        y_full,
                        doDegree,
                        picklePrefix=f"{picklePrefixes[doDegree]}Model.",
                        randomOrder=True,
                        notRecompute=not cliArgs.recompute,
                        maxiter=cliArgs.maxiter,
                        onlyImprove=cliArgs.improve,
                        modelsToProduce=cliArgs.models,
                        showDone=cliArgs.showDone,
                        cvPostfix=f"_CV-{cvsName}_U-{uksName}",
                        modelsPath=cliArgs.modelsPath,
                        sanityCheck=cliArgs.sanityCheck,
                        saveBest=cliArgs.saveBest,
                        progressBar=[(prog, modelProg), (prog, totalProg)],
                )
                prog.advance(splitProg)
                splitsDone += 1
                if not splitsDone == numSplits:
                    prog.reset(modelProg)
            prog.advance(repProg)
            if not prog.finished:
                prog.reset(splitProg)
                prog.reset(modelProg)


def showResults(models, degree, cvSize=0, cvSummary=False, latex=False):
    def tmpsort(x):
        if x[2] is not None:
            return -x[2]
        else:
            return float("inf")

    assert cvSize >= 0
    assert not (cvSize == 0 and cvSummary)
    if cvSize == 0:
        m = models
    else:
        if cvSummary:
            m = [x[0] for x in models]
            cvName = [x[1] for x in models]
        else:
            m, cvName = models
    if not cvSummary:
        m.sort(key=tmpsort)
        if latex:
            data = []
            #
            svrs = [model for model in m if "SVR" in model[0]]
            if len(svrs) > 0:
                svrs.sort(key=tmpsort)
                bestSvr = svrs[0]
                svrs.remove(bestSvr)
                svrs = [s[0] for s in svrs]
                bestSvr = bestSvr[0]
            #
            for d in [["", x[0], x[2]] for x in m]:
                _, name, conf = d
                if name in svrs:
                    continue
                if conf is not None:
                    conf = f"{conf:.4f}"
                if name == "Linear Regression":
                    name = "Ordinary Least Squares"
                name = name.replace("Regression", "")
                name = name.replace("Regressor", "")
                if "SVR" in name:
                    name = "SVR"
                data.append(["", name, conf])
            for model in ["Lasso", "Elastic Net", "Ridge"]:
                b = [x for x in data if x[1] == model]
                c = [x for x in data if x[1] == f"{model} CV"]
                if len(b) == 0 and len(c) == 0:
                    continue
                if len(b) > 0:
                    b = b[0]
                    if b[2] is None:
                        data.remove(b)
                else:
                    b = [None, None, None]
                if len(c) > 0:
                    c = c[0]
                    if c[2] is None:
                        data.remove(c)
                else:
                    c = [None, None, None]
                if b[2] is not None and c[2] is not None:
                    if b[2] > c[2]:
                        data.remove(c)
                        b[1] = model
                    else:
                        data.remove(b)
                        c[1] = model
                if b[2] is None and c[2] is None:
                    data.append(["", model, None])

            if degree == 1:
                data[0][0] = "Linear"
            else:
                data[0][0] = f"Polynomial (degree {degree})"
            fmt = "latex"
            numparse = False
        else:
            data = [["", x[0], f"{x[2]:.4f}" if x[2] is not None else None] for x in m]
            fmt = "simple"
            numparse = False
        if cvSize > 0:
            data[0][0] = (
                    data[0][0] + f"\n{', '.join([re.compile('nfcore/(.*?):.*').match(cv).group(1) for cv in cvName])}"
            )
        tableHeaders = (f"degree={degree}", "model", "confidence")
    else:
        res = {}
        for n in get_model_names(longname=True):
            for i, x in enumerate(m):
                for r in x:
                    name, _, conf, deg, full_conf = r
                    if name == n:
                        if conf is not None and full_conf is not None:
                            gconf = commons.jamGeomean([conf, full_conf])
                        else:
                            gconf = None
                        if name not in res.keys():
                            res[name] = [(gconf, cvName[i])]
                        else:
                            res[name].append((gconf, cvName[i]))
        #
        if latex:
            for model in ["Lasso", "Elastic Net", "Ridge"]:
                if model in res.keys() and model + " CV" in res.keys():
                    r = res[model]
                    rcv = res[model + " CV"]
                    c = 0
                    for cv in cvName:
                        a = [x for x in r if x[1] == cv][0]
                        b = [y for y in rcv if y[1] == cv][0]
                        if a is not None and b is not None:
                            c += int(a[0] >= b[0])
                        elif a is not None and b is None:
                            c += 1

                    a = res.pop(model, None)
                    b = res.pop(model + " CV", None)
                    if c >= math.ceil(len(cvName) / 2):
                        res[model] = a
                    else:
                        res[model] = b
                else:
                    a = res.pop(model, None)
                    b = res.pop(model + " CV", None)
                    if a is not None:
                        res[model] = a
                    elif b is not None:
                        res[model] = b
            #
            svrs = dict()
            for n in list(res.keys()).copy():
                if "SVR" in n:
                    svrs[n] = res.pop(n)
            if len(svrs) > 0:
                svrComp = dict()
                for s in svrs.keys():
                    svr = svrs[s]
                    # prod = functools.reduce(lambda x, y: x * y, [x[0] for x in svr], 1)
                    svrComp[s] = (
                            commons.jamGeomean([x[0] for x in svr if x[0] is not None])
                            if len([x[0] for x in svr if x[0] is not None]) > 0
                            else float("-inf")
                    )
                bestSvr = [svr for n, svr in svrs.items() if svrComp[n] == max([a for a in svrComp.values()])][0]
                res["SVR"] = bestSvr
        #
        data = []
        for n in res.keys():
            for r in res[n]:
                conf, cv = r
                data.append(["", n, conf, cv])
        for i, d in enumerate(data):
            p, name, conf, cv = d
            if conf is not None:
                conf = f"{conf:.4f}"
            if latex:
                if name == "Linear Regression":
                    name = "Ordinary Least Squares"
                name = name.replace("Regression", "")
                name = name.replace("Regressor", "")
            cv = ", ".join([re.compile("nfcore/(.*?):.*").match(c).group(1) for c in cv])
            data[i] = [p, name, conf, cv]
        grouped = {}
        for d in data:
            p, name, conf, cv = d
            if name not in grouped.keys():
                grouped[name] = [d]
            else:
                grouped[name].append(d)
        for n, g in grouped.items():
            g.sort(key=lambda x: x[3])
        grouped = list(grouped.values())
        for j, g in enumerate(grouped):
            v = [float(d[2]) for d in g if d[2] is not None]
            if len(v):
                avg = commons.jamGeomean(v)
            else:
                avg = None
            for i, d in enumerate(g):
                p, name, conf, cv = d
                if i == math.floor(len(g) / 2):
                    g[i] = [
                            p,
                            name,
                            conf,
                            f"{avg:.4f}" if avg is not None else None,
                            cv,
                    ]
                else:
                    g[i] = [p, name, conf, "", cv]
            grouped[j] = g
        grouped.sort(
                key=lambda x: float(x[math.floor(len(g) / 2)][3])
                if x[math.floor(len(g) / 2)][3] is not None
                else float("-inf"),
                reverse=True,
        )
        data = [d for g in grouped for d in g]
        fmt = "latex" if latex else "simple"
        numparse = False
        tableHeaders = (f"degree={degree}", "model", "confidence", "average", "cv fold")
    print(tabulate(data, headers=tableHeaders, tablefmt=fmt, missingval="N/A", disable_numparse=not numparse, ))


class ScistatsNormBetween:
    def __init__(
            self,
            small: float,
            large: float,
            cond: Optional[Callable[[float], bool]] = None,
            div: Optional[float] = 2,
            toint: Optional[bool] = False,
            clip: Optional[Union[bool, Sequence]] = False,
            hardClip: Optional[Union[bool, Sequence]] = False,
            center: Optional[float] = None,
    ) -> None:
        if small <= large:
            self.lower = small
            self.upper = large
        else:
            self.lower = large
            self.upper = small
        clip_cond = lambda x: True
        if isinstance(clip, bool) and clip:
            clip_cond = lambda x: self.lower <= x <= self.upper
        if clip is not None and is_sequence(clip) and len(clip) >= 2:
            clip_cond = lambda x: clip[0] <= x <= clip[1]
        if isinstance(hardClip, bool) and hardClip:
            clip_cond = lambda x: self.lower < x < self.upper
        if hardClip is not None and is_sequence(hardClip) and len(hardClip) >= 2:
            clip_cond = lambda x: hardClip[0] < x < hardClip[1]
        if cond is None:
            self.cond = lambda x: True and clip_cond(x)
        else:
            self.cond = lambda x: cond(x) and clip_cond(x)
        if center is not None:
            assert self.lower <= center <= self.upper, f"center must be within range of [{self.lower}, {self.upper}]"
            self.center = center
        else:
            self.center = (self.upper + self.lower) / 2
        self.norm = scistats.norm(loc=self.center, scale=(self.upper - self.lower) / (div * 2))
        self.toint = toint

    def rvs(self, size: int = 1, *args, **kwargs) -> Union[float, List[float]]:
        if size < 1:
            return []
        if size > 1:
            return [self.rvs(*args, **kwargs) for i in range(size)]
        else:
            tries = 1
            while tries < MAXTRIES:
                r = self.norm.rvs(*args, **kwargs)
                if self.toint:
                    r = commons.iround(r)
                if self.cond(r):
                    break
                tries += 1
            return r


class ScistatsNormAround(ScistatsNormBetween):
    def __init__(
            self,
            center: float,
            dist: float,
            cond: Optional[Callable[[float], bool]] = None,
            div: Optional[float] = 2,
            toint: Optional[bool] = False,
            clip: Optional[bool] = False,
            hardClip: Optional[bool] = False,
    ) -> None:
        super(ScistatsNormAround, self).__init__(center - dist, center + dist, cond, div, toint, clip, hardClip)


class SciStatsNormBetweenRandTuple:
    def __init__(
            self,
            small: float,
            large: float,
            tupleSize: Tuple,
            cond: Optional[Callable[[float], bool]] = None,
            div: Optional[float] = 2,
            toint: Optional[bool] = False,
            clip: Optional[Union[bool, Sequence]] = False,
            hardClip: Optional[Union[bool, Sequence]] = False,
            center: Optional[float] = None,
            maxTotal: Optional[float] = None,
    ):
        self.dist = ScistatsNormBetween(small, large, cond, div, toint, clip, hardClip, center)
        assert is_sequence(tupleSize) and len(tupleSize) >= 2
        if tupleSize[0] <= tupleSize[1]:
            self.tupleSize = (tupleSize[0], tupleSize[1])
        else:
            self.tupleSize = (tupleSize[1], tupleSize[0])
        if maxTotal is not None:
            self.maxTotal = maxTotal
        else:
            self.maxTotal = float("inf")

    def rvs(self, *args, **kwargs):
        if "size" in kwargs.keys():
            kwargs.pop("size")
        randSize = random.randint(self.tupleSize[0], self.tupleSize[1])
        out = self.dist.rvs(randSize, *args, **kwargs)
        tries = 1
        if randSize > 1:
            while sum(out) > self.maxTotal and tries < MAXTRIES:
                out = self.dist.rvs(randSize, *args, **kwargs)
                tries += 1
        else:
            while out > self.maxTotal and tries < MAXTRIES:
                out = self.dist.rvs(randSize, *args, **kwargs)
                tries += 1
        return out


def get_model_names(longname: bool = False) -> List[str]:
    if longname:
        return [x[3] for x in get_models()]
    else:
        return [x[0] for x in get_models()]


def get_models(
        randomOrder: bool = False, maxiter: int = -1, restrict: Optional[List[str]] = None
) -> List[Tuple[str, Any, Union[None, Dict[str, Any]], str, Union[None, Dict[str, Any]]]]:
    if maxiter <= 0:
        maxiterPos = 100_000
    else:
        maxiterPos = maxiter
    models: List[Tuple[str, Any, Union[None, Dict[str, Any]], str, Union[None, Dict[str, Any]]]] = [
            ("Linear", linear_model.LinearRegression(n_jobs=-1), None, "Linear Regression", None,),
            (
                    "SVR-linear",
                    svm.SVR(max_iter=maxiter, cache_size=1000),
                    {
                            "kernel": ["linear"],
                            "C": ScistatsNormBetween(0, 1e2, cond=(lambda x: x > 0)),
                            "epsilon": ScistatsNormBetween(0, 10, cond=(lambda x: x >= 0.1)),
                            "tol": ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True),
                    },
                    "SVR - linear",
                    None,
            ),
            (
                    "SVR-poly",
                    svm.SVR(max_iter=maxiter, cache_size=1000),
                    {
                            "kernel": ["poly"],
                            "degree": [2, 3],
                            "gamma": ScistatsNormBetween(0, 1, cond=(lambda x: x > 0)),
                            "C": ScistatsNormBetween(0, 1e2, cond=(lambda x: x > 0)),
                            "epsilon": ScistatsNormBetween(0, 10, cond=(lambda x: x >= 0.1)),
                            "tol": ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True),
                            "coef0": ScistatsNormAround(0, 10),
                    },
                    "SVR - poly",
                    None,
            ),
            (
                    "SVR-rbf",
                    svm.SVR(max_iter=maxiter, cache_size=1000),
                    {
                            "kernel": ["rbf"],
                            "gamma": ScistatsNormBetween(0, 1, cond=(lambda x: x > 0)),
                            "C": ScistatsNormBetween(0, 1e2, cond=(lambda x: x > 0)),
                            "epsilon": ScistatsNormBetween(0, 10, cond=(lambda x: x >= 0.1)),
                            "tol": ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True),
                    },
                    "SVR - rbf",
                    None,
            ),
            (
                    "SVR-sigmoid",
                    svm.SVR(max_iter=maxiter, cache_size=1000),
                    {
                            "kernel": ["sigmoid"],
                            "gamma": ScistatsNormBetween(0, 1, cond=(lambda x: x > 0)),
                            "C": ScistatsNormBetween(0, 1e2, cond=(lambda x: x > 0)),
                            "epsilon": ScistatsNormBetween(0, 10, cond=(lambda x: x >= 0.1)),
                            "tol": ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True),
                            "coef0": ScistatsNormAround(0, 10),
                    },
                    "SVR - sigmoid",
                    None,
            ),
            (
                    "Lasso",
                    linear_model.Lasso(max_iter=maxiterPos),
                    {
                            "alpha": ScistatsNormAround(1, 10, cond=(lambda x: x >= 0.1)),
                            "tol": ScistatsNormAround(0, 1e-2, cond=(lambda x: x >= 1e-4), clip=True),
                            "selection": ["random"]  # , 'cyclic']
                            # 'warm_start': [True, False]
                    },
                    "Lasso",
                    None,
            ),
            (
                    "LassoCV",
                    linear_model.LassoCV(max_iter=maxiterPos, n_jobs=-1),
                    {
                            "eps": ScistatsNormBetween(1e-4, 1e-2, cond=(lambda x: x > 0)),
                            "n_alphas": ScistatsNormBetween(10, 1000, cond=(lambda x: x >= 10), toint=True),
                            "tol": ScistatsNormAround(0, 1e-2, cond=(lambda x: x >= 1e-4), clip=True),
                            "cv": ScistatsNormBetween(2, 4, clip=True, toint=True),
                            "selection": ["random"]  # , 'cyclic']
                            # 'warm_start': [True, False]
                    },
                    "Lasso CV",
                    {"skipPreElim": True, "cv": 1},
            ),
            (
                    "Ridge",
                    linear_model.Ridge(),
                    {
                            "alpha": ScistatsNormAround(1, 10, cond=(lambda x: x > 0)),
                            "tol": ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True),
                    },
                    "Ridge",
                    None,
            ),
            (
                    "RidgeCV",
                    linear_model.RidgeCV(alphas=ScistatsNormAround(1, 100, cond=(lambda x: x > 0)).rvs(size=5)),
                    {"cv": ScistatsNormBetween(2, 4, clip=True, toint=True)},
                    "Ridge CV",
                    {"skipPreElim": True, "cv": 1},
            ),
            (
                    "ElasticNet",
                    linear_model.ElasticNet(max_iter=maxiterPos),
                    {
                            "alpha": ScistatsNormAround(1, 10, cond=(lambda x: x >= 0.1)),
                            "l1_ratio": ScistatsNormBetween(0, 1, clip=True),
                            "tol": ScistatsNormAround(0, 1e-2, cond=(lambda x: x >= 1e-4), clip=True),
                            "selection": ["random"]  # , 'cyclic']
                            # 'warm_start': [True, False]
                    },
                    "Elastic Net",
                    None,
            ),
            (
                    "ElasticNetCV",
                    linear_model.ElasticNetCV(max_iter=maxiterPos, n_jobs=-1),
                    {
                            "l1_ratio": ScistatsNormBetween(0, 1, clip=True),
                            "eps": ScistatsNormBetween(1e-4, 1e-2, cond=(lambda x: x > 0)),
                            "n_alphas": ScistatsNormBetween(10, 1000, cond=(lambda x: x >= 10), toint=True),
                            "tol": ScistatsNormAround(0, 1e-2, cond=(lambda x: x >= 1e-4), clip=True),
                            "cv": ScistatsNormBetween(2, 4, clip=True, toint=True),
                            "selection": ["random"]  # , 'cyclic']
                            # 'warm_start': [True, False]
                    },
                    "Elastic Net CV",
                    {"skipPreElim": True, "cv": 1},
            ),
            (
                    "BayesianRidge",
                    linear_model.BayesianRidge(),
                    {
                            "n_iter": ScistatsNormAround(300, 200, cond=(lambda x: x >= 100), toint=True),
                            "alpha_1": ScistatsNormAround(0, 1e-3, cond=(lambda x: 0 < x <= 1e-3)),
                            "alpha_2": ScistatsNormAround(0, 1e-3, cond=(lambda x: 0 < x <= 1e-3)),
                            "lambda_1": ScistatsNormAround(0, 1e-3, cond=(lambda x: 0 < x <= 1e-3)),
                            "lambda_2": ScistatsNormAround(0, 1e-3, cond=(lambda x: 0 < x <= 1e-3)),
                            "alpha_init": ScistatsNormBetween(0, 1, cond=(lambda x: 0 <= x <= 1)),
                            "lambda_init": ScistatsNormAround(1, 10, cond=(lambda x: x > 0)),
                            "tol": ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True),
                    },
                    "Bayesian Ridge",
                    None,
            ),
            (
                    "ARD",
                    linear_model.ARDRegression(),
                    {
                            "n_iter": ScistatsNormAround(300, 200, cond=(lambda x: x >= 100), toint=True),
                            "tol": ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True),
                            "alpha_1": ScistatsNormAround(0, 1e-3, cond=(lambda x: 0 < x <= 1e-3)),
                            "alpha_2": ScistatsNormAround(0, 1e-3, cond=(lambda x: 0 < x <= 1e-3)),
                            "lambda_1": ScistatsNormAround(0, 1e-3, cond=(lambda x: 0 < x <= 1e-3)),
                            "lambda_2": ScistatsNormAround(0, 1e-3, cond=(lambda x: 0 < x <= 1e-3)),
                            "threshold_lambda": ScistatsNormBetween(5000, 20000, clip=True),
                    },
                    "Automatic Relevance Determination Regression",
                    None,
            ),
            (
                    "SGD",
                    linear_model.SGDRegressor(max_iter=maxiterPos),
                    {
                            "loss": ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive", ],
                            "penalty": ["l2", "l1", "elasticnet"],
                            "alpha": ScistatsNormAround(0, 1e-1, cond=(lambda x: 0 < x <= 1e-1)),
                            "l1_ratio": ScistatsNormBetween(0, 1, cond=(lambda x: 0 <= x <= 1)),  # only for penalty=elasticnet
                            "epsilon": ScistatsNormBetween(1e-3, 10, cond=(lambda x: x > 0)),
                            # for loss=huber, epsilon_insensitive, squared_epsilon_insensitive
                            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
                            "eta0": ScistatsNormBetween(1e-3, 1e-1, cond=(lambda x: x > 0)),
                            "tol": ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True),
                            # for learning_rate=constant, invscaling, adaptive
                            "power_t": ScistatsNormBetween(0, 1, cond=(lambda x: 0 < x < 1)),  # for learning_rate=invscaling
                    },
                    "Stochastic Gradient Descent",
                    None,
            ),
            (
                    "PA",
                    linear_model.PassiveAggressiveRegressor(max_iter=maxiterPos),
                    {
                            "C": ScistatsNormAround(0, 1e3, cond=(lambda x: x > 0)),
                            "tol": ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True),
                            "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
                            "epsilon": ScistatsNormAround(0.1, 10, cond=(lambda x: x >= 1e-2))
                            # 'warm_start': [True, False]
                    },
                    "Passive Aggressive Regressor",
                    None,
            ),
            (
                    "Huber",
                    linear_model.HuberRegressor(max_iter=maxiterPos),
                    {
                            "epsilon": ScistatsNormAround(1, 10, cond=(lambda x: x > 1)),
                            "alpha": ScistatsNormAround(0, 1e-1, cond=(lambda x: 0 < x <= 1e-1)),
                            "tol": ScistatsNormAround(0, 1e-3, cond=(lambda x: x >= 1e-5), clip=True)
                            # 'warm_start': [True, False]
                    },
                    "Huber Regressor",
                    None,
            ),
            (
                    "TheilSen",
                    linear_model.TheilSenRegressor(
                            n_jobs=-1,
                            max_iter=maxiterPos,
                            tol=ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True).rvs(),
                    ),
                    None,
                    "Theil Sen",
                    None,
            ),
            (
                    "NNR",
                    neural_network.MLPRegressor(max_iter=maxiterPos, ),
                    {
                            "hidden_layer_sizes": SciStatsNormBetweenRandTuple(
                                    50, 150, (1, 3), clip=True, center=100, toint=True, maxTotal=300
                            ),
                            "activation": ["identity", "logistic", "tanh", "relu"],
                            "solver": ["lbfgs", "sgd", "adam"],
                            "alpha": ScistatsNormBetween(10e-7, 10, clip=True, center=1e-1),
                            "learning_rate": ["constant", "invscaling", "adaptive"],
                            "learning_rate_init": ScistatsNormBetween(1e-5, 1e-2, clip=True, center=1e-3),
                            "tol": ScistatsNormBetween(0, 1e-2, clip=True, cond=lambda x: x >= 1e-5),
                            "warm_start": [True, False],
                            "early_stopping": [True, ],  # False
                            "power_t": ScistatsNormBetween(0, 1, hardClip=True, div=3),
                            "momentum": ScistatsNormBetween(0.5, 1, hardClip=True, center=0.9),
                            "nesterovs_momentum": [True, False],
                            "beta_1": ScistatsNormBetween(0.8, 1, hardClip=True, center=0.9),
                            "beta_2": ScistatsNormBetween(0.9, 1, hardClip=True, center=0.999),
                    },
                    "Neural Network Regressor",
                    {"minBaseRes": 40, "minCand": 100, "cv": 1, "maxNumTurns": 10, "maxPrelim": 3},
            ),
            (
                    "NNC",
                    neural_network.MLPClassifier(max_iter=maxiterPos, ),
                    {
                            "hidden_layer_sizes": SciStatsNormBetweenRandTuple(
                                    50, 150, (1, 3), clip=True, center=100, toint=True, maxTotal=300
                            ),
                            "activation": ["identity", "logistic", "tanh", "relu"],
                            "solver": ["lbfgs", "sgd", "adam"],
                            "alpha": ScistatsNormBetween(10e-7, 10, clip=True, center=1e-1),
                            "learning_rate": ["constant", "invscaling", "adaptive"],
                            "learning_rate_init": ScistatsNormBetween(1e-5, 1e-2, clip=True, center=1e-3),
                            "tol": ScistatsNormBetween(0, 1e-2, clip=True, cond=lambda x: x >= 1e-5),
                            "warm_start": [True, False],
                            "early_stopping": [True, ],  # False
                            "power_t": ScistatsNormBetween(0, 1, hardClip=True, div=3),
                            "momentum": ScistatsNormBetween(0.5, 1, hardClip=True, center=0.9),
                            "nesterovs_momentum": [True, False],
                            "beta_1": ScistatsNormBetween(0.8, 1, hardClip=True, center=0.9),
                            "beta_2": ScistatsNormBetween(0.9, 1, hardClip=True, center=0.999),
                    },
                    "Neural Network Classifier",
                    {"minBaseRes": 40, "minCand": 100, "cv": 1, "maxNumTurns": 10, "maxPrelim": 3},
            ),
            (
                    "DTR",
                    tree.DecisionTreeRegressor(),
                    {
                            "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                            "splitter": ["best", "random"],
                            "max_features": ["auto", "sqrt", "log2"],
                    },
                    "Decision Tree Regressor",
                    None,
            ),
            (
                    "DTC",
                    tree.DecisionTreeClassifier(),
                    {
                            "criterion": ["gini", "entropy"],
                            "splitter": ["best", "random"],
                            "max_features": ["auto", "sqrt", "log2"],
                    },
                    "Decision Tree Classifier",
                    None,
            ),
    ]
    if restrict is not None:
        models = [x for x in models if x[0] in restrict]
    if randomOrder:
        random.shuffle(models)
    return models


def getHRSCVTournamentParams(params, halvingParams, X_train):
    fact, numCand, baseRes, cv = _getHRSCVTournamentParams(params, halvingParams, X_train)
    best = (fact, numCand, baseRes, cv)
    if halvingParams is not None:
        if "minCand" in halvingParams.keys():
            h_minCand = halvingParams["minCand"]
            tries = 1
            bestCand = numCand
            while numCand < h_minCand and tries < MAXTRIES:
                fact, numCand, baseRes, cv = _getHRSCVTournamentParams(params, halvingParams, X_train)
                if numCand >= bestCand:
                    best = (fact, numCand, baseRes, cv)
                    bestCand = numCand
                tries += 1
    return best


def _getHRSCVTournamentParams(params, halvingParams, X_train):
    cv = ScistatsNormBetween(2, 4, cond=(lambda x: 2 <= x <= 5), toint=True).rvs()
    regrcv = ScistatsNormBetween(1, 1)  # this is pretty ugly and only needed because of the baseRes cond
    minBaseRes = 8
    maxBaseRes = commons.iround(0.02 * len(X_train))
    minNumTurns = max(cv / 2, 3)
    maxNumTurns = min(4 * cv, 8)
    minPrelim = 0
    maxPrelim = cv / 2
    if params is not None:
        if "cv" in params.keys():
            regrcv = params["cv"]
    if halvingParams is not None:
        if "cv" in halvingParams.keys():
            cv = halvingParams["cv"]
            if cv is None or cv <= 1:
                cv = 2
        if "minNumTurns" in halvingParams.keys():
            minNumTurns = halvingParams["minNumTurns"]
        if "maxNumTurns" in halvingParams.keys():
            maxNumTurns = halvingParams["maxNumTurns"]
        if "minPrelim" in halvingParams.keys():
            minPrelim = halvingParams["minPrelim"]
        if "maxPrelim" in halvingParams.keys():
            maxPrelim = halvingParams["maxPrelim"]
        if "minBaseRes" in halvingParams.keys():
            h_minBaseRes = halvingParams["minBaseRes"]
            if 0 < h_minBaseRes < 1:
                minBaseRes = commons.iround(h_minBaseRes * len(X_train))
            else:
                minBaseRes = h_minBaseRes
        if "maxBaseRes" in halvingParams.keys():
            h_maxBaseRes = halvingParams["maxBaseRes"]
            if 0 < h_maxBaseRes < 1:
                maxBaseRes = commons.iround(h_maxBaseRes * len(X_train))
            else:
                maxBaseRes = h_maxBaseRes
    baseRes = ScistatsNormBetween(
            minBaseRes,
            maxBaseRes if maxBaseRes >= minBaseRes else minBaseRes,
            cond=(lambda x: x > 2 * cv * regrcv.upper),
            toint=True,
    ).rvs()  # TODO: x>=9 is kind of arbitrary, whenever the regr also does CV, x must be bigger than 2*cv*<cv of regr> --> for x>=9 this should be the case but not all models require x>=9 so its a dirty fix for now
    numTurns = ScistatsNormBetween(
            minNumTurns, maxNumTurns, clip=True, toint=True, center=minNumTurns + 1 / 3 * (maxNumTurns - minNumTurns)
    ).rvs()
    prelimRounds = ScistatsNormBetween(minPrelim, maxPrelim, clip=True, toint=True).rvs()
    if halvingParams is not None:
        if "skipPreElim" in halvingParams.keys() and halvingParams["skipPreElim"]:
            prelimRounds = 0
    # if numTurns >= 15:
    #     prelimRounds = 0
    lastRoundResPercent = ScistatsNormBetween(0.8, 1.0, clip=True, center=0.95).rvs()
    lastRoundRes = commons.iround(len(X_train) * lastRoundResPercent)
    fact = (lastRoundRes / baseRes) ** (1 / numTurns)
    lastRoundNumCand = ScistatsNormBetween(1, max(cv, 2), clip=True, toint=True).rvs()
    numCand = commons.iround(lastRoundNumCand * (fact ** (numTurns + prelimRounds - 1)))
    return fact, numCand, baseRes, cv


def sanity_check(
        pFile: Union[Path, str],
        X_train: List[List[float]],
        y_train: List[List[float]],
        X_test: List[List[float]],
        y_test: List[List[float]],
        X_unknown: List[List[float]],
        y_unknown: List[List[float]],
        X_full: List[List[float]],
        y_full: List[List[float]],
        bonusPickleInfo: Optional[Any] = None,
        loaded: Optional[PickleOut] = None,
) -> PickleOut:
    if loaded is None:
        loaded: PickleOut = pickle.load(open(pFile, "br"))
    try:
        (
                longname,
                regr,
                test_confidence,
                deg,
                full_confidence,
                unknown_confidence,
                train_confidence,
                bonusPickleInfo,
        ) = loaded
    except:
        longname, regr, test_confidence, deg, full_conf = loaded
    train_confidence = regr.score(X_train, y_train)
    test_confidence = regr.score(X_test, y_test)
    full_confidence = regr.score(X_full, y_full)
    unknown_confidence = regr.score(X_unknown, y_unknown) if len(X_unknown) > 0 else None
    toDump = PickleOut(
            longname, regr, test_confidence, deg, full_confidence, unknown_confidence, train_confidence, bonusPickleInfo,
    )
    with open(pFile, "bw") as f:
        pickle.dump(toDump, f)
    return toDump


def fit_models(
        X_train: List[List[float]],
        y_train: List[List[float]],
        X_test: List[List[float]],
        y_test: List[List[float]],
        X_unknown: List[List[float]],
        y_unknown: List[List[float]],
        X_full: List[List[float]],
        y_full: List[List[float]],
        polyDeg: int,
        models: Optional[List[Tuple[str, Any, Union[None, Dict[str, Any]], str, Union[None, Dict[str, Any]]]]] = None,
        picklePrefix: Optional[str] = "",
        randomOrder: Optional[bool] = False,
        notRecompute: Optional[bool] = True,
        maxiter: Optional[int] = -1,
        onlyImprove: Optional[bool] = False,
        modelsToProduce: Optional[List[str]] = None,
        showDone: Optional[bool] = False,
        cvPostfix: Optional[str] = None,
        modelsPath: Union[Path, str] = None,
        sanityCheck: Optional[bool] = False,
        saveBest: Optional[bool] = False,
        bonusPickleInfo: Optional[Any] = None,
        progressBar: Optional[Union[Tuple[rich.progress.Progress, int], List[Tuple[rich.progress.Progress, int]]]] = None,
) -> List[PickleOut]:
    # TODO: fix this https://stats.stackexchange.com/questions/431883/lasso-regression-doesnt-converge-in-case-of-zero-y-vector in sklearn
    def custom_scoring(est: sklearn.base.RegressorMixin, X: List[List[float]], y: List[List[float]]) -> float:
        portion = len(X) / len(X_train)
        portion = math.sqrt(portion)
        if portion < 0.05:
            portion = 0.05
        if portion > 0.8:
            portion = 0.8
        #
        test_portion = commons.iround(portion * len(X_test))
        if test_portion > len(X_test):
            test_portion = len(X_test)
        if test_portion < 2:
            test_portion = 2
        full_portion = commons.iround(portion * len(X_full))
        if full_portion > len(X_full):
            full_portion = len(X_full)
        if full_portion < 2:
            full_portion = 2
        test_samples = random.sample(range(len(X_test)), test_portion)
        full_samples = random.sample(range(len(X_full)), full_portion)
        train_confidence = est.score(X, y)
        test_confidence = est.score(
                [t for i, t in enumerate(X_test) if i in test_samples],
                [t for i, t in enumerate(y_test) if i in test_samples],
        )
        full_confidence = est.score(
                [t for i, t in enumerate(X_full) if i in full_samples],
                [t for i, t in enumerate(y_full) if i in full_samples],
        )
        newGeo = commons.jamGeomean([test_confidence, full_confidence, train_confidence])
        return newGeo

    #
    if modelsPath is None:
        modelsPath = "./models"
    if cvPostfix is None:
        cvPostfix = ""
    if modelsToProduce is None:
        modelsToProduce = get_model_names()
    if models is None:
        models: list[tuple[str, Any, Optional[dict[str, Any]], str, Optional[dict[str, Any]]]] = get_models(
                randomOrder=randomOrder, maxiter=maxiter, restrict=modelsToProduce
        )
    # printInfo(f"Processing models: {[x[3] for x in models]}")
    trained: List[PickleOut] = []
    for model in models:
        modelName, regr, params, longname, halvingParams = model
        if params is not None:
            fact, numCand, baseRes, cv = getHRSCVTournamentParams(params, halvingParams, X_train)
            searchParams = {
                    "estimator": regr,
                    "param_distributions": params,
                    "n_jobs": -1,
                    "factor": fact,
                    "n_candidates": numCand,
                    "min_resources": baseRes,
                    "aggressive_elimination": True,
                    "refit": True,
                    # 'return_train_score'    : True,
                    "scoring": custom_scoring,
                    "error_score": -1e-50,
                    "cv": cv,
                    "verbose": 1,
            }
            regr = HalvingRandomSearchCV(**searchParams)
            # regr = RandomizedSearchCV(estimator=regr, param_distributions=params, n_iter=2 ** 8, n_jobs=-1, refit=True,
            #                           return_train_score=True, verbose=2, cv=10)
        pickleName = f"{picklePrefix}{modelName}{cvPostfix}"
        fullname = f"{picklePrefix}{longname}"
        # spinner = Halo(text=fullName + ':  ', spinner='dots')
        pFile = Path(f"{modelsPath}/{pickleName}.pickle")
        # spinner.start()
        if not showDone and (not pFile.is_file() or not notRecompute or onlyImprove):
            printInfo(f"{fullname}:")
            if not pFile.is_file():
                printInfo("Found no pickle", 1)
            if not notRecompute:
                printInfo("Recomputing", 1)
            try:
                regr.fit(X_train, y_train)
                train_confidence = regr.score(X_train, y_train)
                test_confidence = regr.score(X_test, y_test)
                full_confidence = regr.score(X_full, y_full)
                unknown_confidence = regr.score(X_unknown, y_unknown) if len(X_unknown) > 0 else None
                if params is not None:
                    # toDump = (longname, regr.best_estimator_, test_confidence, polyDeg, full_confidence)
                    toDump = PickleOut(
                            longname,
                            regr.best_estimator_,
                            test_confidence,
                            polyDeg,
                            full_confidence,
                            unknown_confidence,
                            train_confidence,
                            bonusPickleInfo,
                    )
                else:
                    # toDump = (longname, regr, test_confidence, polyDeg, full_confidence)
                    toDump = PickleOut(
                            longname,
                            regr,
                            test_confidence,
                            polyDeg,
                            full_confidence,
                            unknown_confidence,
                            train_confidence,
                            bonusPickleInfo,
                    )
                if onlyImprove:
                    printInfo(f"Trying to improve {fullname}", 1)
                    if not pFile.is_file():
                        printWarn(
                                f"There was no pickle found for {fullname} (at {pickleName}.pickle) to improve", 1,
                        )
                        pickle.dump(toDump, open(pFile, "bw"))
                        trained.append(toDump)
                    else:
                        loaded = pickle.load(open(pFile, "br"))
                        if sanityCheck:
                            loaded = sanity_check(
                                    pFile,
                                    X_train,
                                    y_train,
                                    X_test,
                                    y_test,
                                    X_unknown,
                                    y_unknown,
                                    X_full,
                                    y_full,
                                    bonusPickleInfo,
                                    loaded,
                            )
                        newGeo = commons.jamGeomean([test_confidence, full_confidence, train_confidence])
                        loadedGeo = commons.jamGeomean(
                                [loaded.test_confidence, loaded.full_confidence, loaded.train_confidence, ]
                        )
                        table = tabulate(
                                [
                                        [
                                                "old:",
                                                loaded.test_confidence,
                                                loaded.full_confidence,
                                                loaded.train_confidence,
                                                loadedGeo,
                                        ],
                                        ["new:", test_confidence, full_confidence, train_confidence, newGeo, ],
                                ],
                                headers=["", "test_conf", "full_conf", "train_conf", "geo_mean", ],
                        )
                        geoDiff = newGeo - loadedGeo
                        geoPercent = (100 * newGeo / loadedGeo) - 100
                        if newGeo >= loadedGeo:
                            pickle.dump(toDump, open(pFile, "bw"))
                            printInfo(
                                    f"Improved (or matched) {fullname} (over {pickleName}.pickle): (+{geoDiff:.5f}, +{geoPercent:.5f}%)",
                                    1,
                            )
                            trained.append(toDump)
                        else:
                            printInfo(
                                    f"Did not improve {fullname} (over {pickleName}.pickle): ({geoDiff:.5f}, {geoPercent:.5f}%)", 1,
                            )
                            trained.append(loaded)
                        print("\n".join(["\t" + row for row in table.split("\n")]))
                else:
                    pickle.dump(toDump, open(pFile, "bw"))
                    printInfo(f"Saved {fullname} (to {pickleName}.pickle)", 1)
                    trained.append(toDump)
                # spinner.succeed(f"Finished fitting {fullName}")
                printSucc(f"Finished fitting {fullname}", 1)
            except Exception as e:
                # spinner.fail("failed")
                printError(f"Failed to fit {fullname} with error:", 1)
                rc.print_exception(show_locals=False)
                exit(1)
        else:
            if showDone:
                printInfo(f"{fullname}:")
                if not pFile.is_file():
                    printInfo("Found no pickle", 1)
                    trained.append(PickleOut(longname, None, None, polyDeg, None, None, None, None))
                else:
                    printInfo("Found pickle", 1)
                    loaded = pickle.load(open(pFile, "br"))
                    if sanityCheck:
                        loaded = sanity_check(
                                pFile,
                                X_train,
                                y_train,
                                X_test,
                                y_test,
                                X_unknown,
                                y_unknown,
                                X_full,
                                y_full,
                                bonusPickleInfo,
                                loaded,
                        )
                    trained.append(loaded)
                    printSucc("Sanity checked", 1)
            else:
                printInfo(f"{fullname}:")
                loaded = pickle.load(open(pFile, "br"))
                if sanityCheck:
                    loaded = sanity_check(
                            pFile,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            X_unknown,
                            y_unknown,
                            X_full,
                            y_full,
                            bonusPickleInfo,
                            loaded,
                    )
                trained.append(loaded)
                # spinner.succeed(f"Found pickle for {fullName}")
                printSucc(f"Found pickle for {fullname} at {pickleName}.pickle", 1)
        if progressBar is not None:
            if type(progressBar) is list:
                for p in progressBar:
                    prog, bar = p
                    prog.advance(bar)
            else:
                prog, bar = progressBar
                prog.advance(bar)
    #
    if saveBest:
        bestModelPath = Path(f"{modelsPath}/bestModel.pickle")
        if bestModelPath.exists():
            bestModel = list(pickle.load(open(bestModelPath, "br")))
        else:
            bestModel = list(trained[0])
        for t in trained:
            if t[2] is not None:
                if t[2] > bestModel[2]:
                    bestModel = list(t)
        try:
            name, loadedRegr, conf, deg = bestModel
        except:
            name, loadedRegr, conf = bestModel
            deg = polyDeg
        pickle.dump((name, loadedRegr, conf, deg), open(f"{modelsPath}/bestModel.pickle", "bw"))
    #
    return trained


if __name__ == "__main__":
    main()
