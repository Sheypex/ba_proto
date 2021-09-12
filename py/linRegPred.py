import functools
import itertools
import math
import re
import statistics
from functools import reduce
from pprint import pprint

from alive_progress import alive_bar
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
from sklearn.model_selection import HalvingGridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV
import scipy.stats as scistats
from data_types import PickleOut

from db_actions import db_actions
import numpy as np

TrainTestUnknownSplit = namedtuple("TrainTestUnknownSplit", ["X_train", "y_train", "X_test", "y_test", "X_unknown", "y_unknown", "X_trans", "ukwfs", "cvwfs"])


def printStat(txt, stat=LogSymbols.INFO.value, indent=0):
    if indent > 0:
        ind = ["  "] * indent
        ind = "".join(ind)
    else:
        ind = ""
    print(f"{ind}{stat} {txt}")


def printSucc(txt, indent=0):
    printStat(txt, LogSymbols.SUCCESS.value, indent)


def printInfo(txt, indent=0):
    printStat(txt, LogSymbols.INFO.value, indent)


def printError(txt, indent=0):
    printStat(txt, LogSymbols.ERROR.value, indent)


def printWarn(txt, indent=0):
    printStat(txt, LogSymbols.WARNING.value, indent)


def printBox(message, borderTopBot='#', borderSides='#'):
    print(f"{borderTopBot * (len(message) + 4)}")
    print(f'{borderSides} ' + message + f' {borderSides}')
    print(f"{borderTopBot * (len(message) + 4)}")


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


def getTransformers(X, polyDeg):
    scale = preprocessing.StandardScaler().fit(X)
    if polyDeg > 1:
        poly = preprocessing.PolynomialFeatures(degree=polyDeg, interaction_only=True).fit(X)
    else:
        poly = preprocessing.PolynomialFeatures(degree=polyDeg, interaction_only=True, include_bias=False).fit(X)
    scale2 = preprocessing.StandardScaler().fit(poly.transform(scale.transform(X)))
    return scale, poly, scale2


def applyTransformers(X, scale, poly, scale2):
    return scale2.transform(poly.transform(scale.transform(X)))


def getNumSplits(dF, cvSize=0, unknownSize=0, wfs=None):
    if wfs is None:
        wfs = dF.wfName.unique()
    nSplits = 0
    #
    unknownCombs = [(list(ukwfs), [wf for wf in wfs if wf not in ukwfs]) for ukwfs in itertools.combinations(wfs, unknownSize)]
    for ukSplit in unknownCombs:
        ukwfs, kwfs = ukSplit
        #
        cvCombs = [(list(cvwfs), [wf for wf in kwfs if wf not in cvwfs]) for cvwfs in itertools.combinations(kwfs, cvSize)]
        for cvSplit in cvCombs:
            nSplits += 1
    return nSplits


def getSplits(dF, polyDeg, x_cols, y_cols, cvSize=0, unknownSize=0, wfs=None, randomOrder=True):
    assert 1 <= polyDeg <= 5
    assert 0 <= cvSize <= 4
    assert 0 <= unknownSize <= 4
    assert 0 <= cvSize + unknownSize <= 4
    #
    if wfs is None:
        wfs = dF.wfName.unique()
    #
    splits = list()
    #
    unknownCombs = [(list(ukwfs), [wf for wf in wfs if wf not in ukwfs]) for ukwfs in itertools.combinations(wfs, unknownSize)]
    for ukSplit in unknownCombs:
        ukwfs, kwfs = ukSplit
        #
        cvCombs = [(list(cvwfs), [wf for wf in kwfs if wf not in cvwfs]) for cvwfs in itertools.combinations(kwfs, cvSize)]
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
        rows_known = dF.query("wfName in @kwfs")
        rows_unknown = dF.query("wfName in @ukwfs")
        X_trans = getTransformers(rows_known[x_cols], polyDeg)
        X_scale, X_poly, X_scale2 = X_trans
        #
        X_train = applyTransformers(rows_known.query("wfName in @trainwfs")[x_cols], X_scale, X_poly, X_scale2)
        X_test = X_train if len(cvwfs) == 0 else applyTransformers(rows_known.query("wfName in @cvwfs")[x_cols], X_scale, X_poly, X_scale2)
        X_unknown = list() if len(ukwfs) == 0 else applyTransformers(rows_unknown[x_cols], X_scale, X_poly, X_scale2)
        #
        y_train = rows_known.query("wfName in @trainwfs")[y_cols]
        y_test = y_train if len(cvwfs) == 0 else rows_known.query("wfName in @cvwfs")[y_cols]
        y_unknown = list() if len(ukwfs) == 0 else rows_unknown[y_cols]
        #
        yield TrainTestUnknownSplit(X_train, y_train, X_test, y_test, X_unknown, y_unknown, X_trans, ukwfs, cvwfs)


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('polyDeg', type=int, choices=[1, 2, 3, 4, 5], action='store', default=1)
    argp.add_argument('--csv', type=str, action='store', dest='csvPath')
    argp.add_argument('--recompute', action='store_true', dest='recompute', default=False)
    argp.add_argument('--improve', action='store_true', dest='improve', default=False)
    argp.add_argument('--maxiter', type=int, action='store', dest='maxiter', default=-1)
    argp.add_argument('models', action='store', nargs='*', default='all',
                      choices=['all'] + get_model_names() + [f"_{x}" for x in get_model_names()])
    argp.add_argument('--repeat', action='store', type=int, dest='numRepeats', default=1)
    argp.add_argument('--latex', action='store_true', dest='latex', default=False)
    argp.add_argument('--showDone', action='store_true', dest='showDone', default=False)
    argp.add_argument('--models', action='store', dest='modelsPath', default='models')
    argp.add_argument('--WFCV', action='store', type=int, choices=[0, 1, 2, 3, 4], default=0, dest='cvSize')
    argp.add_argument('--WFU', action='store', type=int, choices=[0, 1, 2, 3, 4], default=0, dest='unknownSize')
    argp.add_argument('--sanity-check', action='store_true', dest='sanityCheck', default=False)
    argp.add_argument('--saveBest', action='store_true', dest='saveBest', default=False)
    cliArgs = argp.parse_args(sys.argv[1:])
    doDegree = cliArgs.polyDeg
    doPoly = doDegree > 1
    assert not (
            cliArgs.recompute and cliArgs.improve), '--recompute and --improve are mutually exclusive'  # mutually exclusive
    if type(cliArgs.models) is str:  # in case only one model or 'all' is given make sure its still in the format of a list
        cliArgs.models = [cliArgs.models]
    modelsToProduce = set(cliArgs.models)  # parse 'all'
    if 'all' in modelsToProduce:
        modelsToProduce.remove('all')
        modelsToProduce = modelsToProduce.union(set(get_model_names()))
    modelsToExclude = []
    for model in modelsToProduce:  # parse excluded models
        if model[0] == '_':  # if model is excluded
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
            f"The path {p} exists but is not a directory. It is an invalid location for storing models.")
    assert cliArgs.cvSize + cliArgs.unknownSize <= 4, f"The sum of workflows used in the cv folds and tested as unknowns may not exceed 4: {cliArgs.cvSize=} and {cliArgs.unknownSize=} were supplied."
    # print cli config
    # printInfo(f"Running with command line arguments: {cliArgs.__dict__}")
    #

    if cliArgs.csvPath:
        dF = None
        try:
            dF = pds.read_csv(open(cliArgs.csvPath, 'r'))  # already filtered for realtime > 1000)
        except:
            pass
        try:
            dF = pds.read_csv(open(cliArgs.csvPath + '.csv', 'r'))
        except:
            pass
        if dF is None:
            raise Exception(f"Couldn't open csv file {cliArgs.csvPath!r} or {(cliArgs.csvPath + '.csv')!r}")
    else:
        with db_actions.connect() as conn:
            dF = pds.read_sql("SELECT * FROM \"averageRuntimesPredictionBase1000\"", conn)
    # print(dF)
    x_cols = ["build-linux-kernel1",
              "fio2", "fio3", "fio4", "fio5", "fio6", "fio7", "fio8", "fio9",
              "iperf10", "iperf11", "iperf12", "iperf13",
              "john-the-ripper14", "john-the-ripper15",
              "ramspeed16", "ramspeed17", "ramspeed18", "ramspeed19", "ramspeed20", "ramspeed21", "ramspeed22", "ramspeed23", "ramspeed24", "ramspeed25",
              "stream26", "stream27", "stream28", "stream29",
              "pCpu", "cpus", "rss", "vmem", "rchar", "wchar", "syscr", "syscw"]
    y_cols = 'rank'
    #
    wfs = dF.wfName.unique()
    wfShortnamesLUT = {wf: re.compile("nfcore/(\w+):.*").match(wf).group(1) for wf in wfs}
    wfLongnamesLUT = {short: long for long, short in wfShortnamesLUT.items()}
    picklePrefixes = {1: 'lin', 2: 'quad', 3: 'cube', 4: 'tet', 5: 'pen'}
    #
    X = dF[x_cols]
    y = dF[y_cols]
    full_scale, full_poly, full_scale2 = getTransformers(X, doDegree)
    X_full = applyTransformers(X, full_scale, full_poly, full_scale2)
    y_full = y
    #
    numSplits = getNumSplits(dF, cliArgs.cvSize, cliArgs.unknownSize, wfs)
    with alive_bar(cliArgs.numRepeats * numSplits * len(cliArgs.models), f"Fitting {len(cliArgs.models)} models on {numSplits} splits for {cliArgs.numRepeats} repeats",
                   enrich_print=False) as progressBar:
        for iReps in range(cliArgs.numRepeats):
            printBox(f"Repeat number {iReps + 1}")
            splits = getSplits(dF, doDegree, x_cols, y_cols, cliArgs.cvSize, cliArgs.unknownSize, wfs)
            for split in splits:
                X_train, y_train, X_test, y_test, X_unknown, y_unknown, X_trans, ukwfs, cvwfs = split
                assert len(X_train) == len(y_train)
                assert len(X_test) == len(y_test)
                assert len(X_unknown) == len(y_unknown)
                printBox(f"{ukwfs=}, {cvwfs=}: {len(X_train)=}, {len(X_test)=}, {len(X_unknown)=}", '-', '|')
                if len(cvwfs) > 0:
                    cvsName = "+".join([wfShortnamesLUT[cv] for cv in cvwfs])
                else:
                    cvsName = "None"
                if len(ukwfs) > 0:
                    uksName = "+".join([wfShortnamesLUT[cv] for cv in ukwfs])
                else:
                    uksName = "None"
                doModels = fit_models(X_train, y_train, X_test, y_test, X_unknown, y_unknown, X_full, y_full, doDegree, picklePrefix=f"{picklePrefixes[doDegree]}Model.",
                                      randomOrder=True,
                                      notRecompute=not cliArgs.recompute, maxiter=cliArgs.maxiter, onlyImprove=cliArgs.improve, modelsToProduce=cliArgs.models,
                                      showDone=cliArgs.showDone, cvPostfix=f"_CV-{cvsName}_U-{uksName}",
                                      modelsPath=cliArgs.modelsPath, sanityCheck=cliArgs.sanityCheck,
                                      saveBest=cliArgs.saveBest, progressBar=progressBar)


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
            for d in [['', x[0], x[2]] for x in m]:
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
                data.append(['', name, conf])
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
                    data.append(['', model, None])

            if degree == 1:
                data[0][0] = "Linear"
            else:
                data[0][0] = f"Polynomial (degree {degree})"
            fmt = 'latex'
            numparse = False
        else:
            data = [['', x[0], f"{x[2]:.4f}" if x[2] is not None else None] for x in m]
            fmt = 'simple'
            numparse = False
        if cvSize > 0:
            data[0][0] = data[0][0] + f"\n{', '.join([re.compile('nfcore/(.*?):.*').match(cv).group(1) for cv in cvName])}"
        tableHeaders = (f"degree={degree}", 'model', 'confidence')
    else:
        res = {}
        for n in get_model_names(longname=True):
            for i, x in enumerate(m):
                for r in x:
                    name, _, conf, deg, full_conf = r
                    if name == n:
                        if conf is not None and full_conf is not None:
                            gconf = jamGeomean([conf, full_conf])
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
                    svrComp[s] = jamGeomean([x[0] for x in svr if x[0] is not None]) if len([x[0] for x in svr if x[0] is not None]) > 0 else float("-inf")
                bestSvr = [svr for n, svr in svrs.items() if svrComp[n] == max([a for a in svrComp.values()])][0]
                res["SVR"] = bestSvr
        #
        data = []
        for n in res.keys():
            for r in res[n]:
                conf, cv = r
                data.append(['', n, conf, cv])
        for i, d in enumerate(data):
            p, name, conf, cv = d
            if conf is not None:
                conf = f"{conf:.4f}"
            if latex:
                if name == "Linear Regression":
                    name = "Ordinary Least Squares"
                name = name.replace("Regression", "")
                name = name.replace("Regressor", "")
            cv = ', '.join([re.compile('nfcore/(.*?):.*').match(c).group(1) for c in cv])
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
                avg = jamGeomean(v)
            else:
                avg = None
            for i, d in enumerate(g):
                p, name, conf, cv = d
                if i == math.floor(len(g) / 2):
                    g[i] = [p, name, conf, f"{avg:.4f}" if avg is not None else None, cv]
                else:
                    g[i] = [p, name, conf, '', cv]
            grouped[j] = g
        grouped.sort(key=lambda x: float(x[math.floor(len(g) / 2)][3]) if x[math.floor(len(g) / 2)][3] is not None else float('-inf'), reverse=True)
        data = [d for g in grouped for d in g]
        fmt = 'latex' if latex else 'simple'
        numparse = False
        tableHeaders = (f"degree={degree}", 'model', 'confidence', 'average', 'cv fold')
    print(tabulate(data, headers=tableHeaders,
                   tablefmt=fmt, missingval='N/A', disable_numparse=not numparse))


class ScistatsNormBetween():
    def __init__(self, small, large, cond=None, div=2, toint=False, clip=False, hardClip=False):
        if clip:
            if hardClip:
                clip_cond = lambda x: small < x < large
            else:
                clip_cond = lambda x: small <= x <= large
        else:
            clip_cond = lambda x: True
        if cond is None:
            self.cond = lambda x: True and clip_cond(x)
        else:
            self.cond = lambda x: cond(x) and clip_cond(x)
        self.norm = scistats.norm(loc=(large + small) / 2, scale=(large - small) / (div * 2))
        self.toint = toint

    def rvs(self, size=1, *args, **kwargs):
        if size > 1:
            return [self.rvs(*args, **kwargs) for i in range(size)]
        else:
            while True:
                r = self.norm.rvs(*args, **kwargs)
                if self.toint:
                    r = iround(r)
                if self.cond(r):
                    break
            return r


class ScistatsNormAround(ScistatsNormBetween):
    def __init__(self, center, dist, cond=None, div=2, toint=False, clip=False, hardClip=False):
        super(ScistatsNormAround, self).__init__(center - dist, center + dist, cond, div, toint, clip, hardClip)


def get_model_names(longname=False):
    if longname:
        return [x[3] for x in get_models()]
    else:
        return [x[0] for x in get_models()]


def get_models(randomOrder=False, maxiter=-1, restrict=None):
    if maxiter <= 0:
        maxiterPos = 100000
    else:
        maxiterPos = maxiter
    models = [
        ("Linear", linear_model.LinearRegression(n_jobs=-1), None,
         "Linear Regression", None),
        ("SVR-linear", svm.SVR(max_iter=maxiter, cache_size=1000),
         {
             'kernel' : ['linear'],
             'C'      : ScistatsNormBetween(0, 1e2, cond=(lambda x: x > 0)),
             'epsilon': ScistatsNormBetween(0, 10, cond=(lambda x: x >= 0.1)),
             'tol'    : ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True)
         },
         "SVR - linear", None),
        ("SVR-poly", svm.SVR(max_iter=maxiter, cache_size=1000),
         {
             'kernel' : ['poly'],
             'degree' : [2, 3],
             'gamma'  : ScistatsNormBetween(0, 1, cond=(lambda x: x > 0)),
             'C'      : ScistatsNormBetween(0, 1e2, cond=(lambda x: x > 0)),
             'epsilon': ScistatsNormBetween(0, 10, cond=(lambda x: x >= 0.1)),
             'tol'    : ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True),
             'coef0'  : ScistatsNormAround(0, 10)
         },
         "SVR - poly", None),
        ("SVR-rbf", svm.SVR(max_iter=maxiter, cache_size=1000),
         {
             'kernel' : ['rbf'],
             'gamma'  : ScistatsNormBetween(0, 1, cond=(lambda x: x > 0)),
             'C'      : ScistatsNormBetween(0, 1e2, cond=(lambda x: x > 0)),
             'epsilon': ScistatsNormBetween(0, 10, cond=(lambda x: x >= 0.1)),
             'tol'    : ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True)
         },
         "SVR - rbf", None),
        ("SVR-sigmoid", svm.SVR(max_iter=maxiter, cache_size=1000),
         {
             'kernel' : ['sigmoid'],
             'gamma'  : ScistatsNormBetween(0, 1, cond=(lambda x: x > 0)),
             'C'      : ScistatsNormBetween(0, 1e2, cond=(lambda x: x > 0)),
             'epsilon': ScistatsNormBetween(0, 10, cond=(lambda x: x >= 0.1)),
             'tol'    : ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True),
             'coef0'  : ScistatsNormAround(0, 10)
         },
         "SVR - sigmoid", None),
        ("Lasso", linear_model.Lasso(max_iter=maxiterPos),
         {
             'alpha'    : ScistatsNormAround(1, 10, cond=(lambda x: x >= 0.1)),
             'tol'      : ScistatsNormAround(0, 1e-2, cond=(lambda x: x >= 1e-4), clip=True),
             'selection': ['random']  # , 'cyclic']
             # 'warm_start': [True, False]
         },
         "Lasso", None),
        ("LassoCV", linear_model.LassoCV(max_iter=maxiterPos, n_jobs=-1),
         {
             'eps'      : ScistatsNormBetween(1e-4, 1e-2, cond=(lambda x: x > 0)),
             'n_alphas' : ScistatsNormBetween(10, 1000, cond=(lambda x: x >= 10), toint=True),
             'tol'      : ScistatsNormAround(0, 1e-2, cond=(lambda x: x >= 1e-4), clip=True),
             'cv'       : ScistatsNormBetween(2, 4, clip=True, toint=True),
             'selection': ['random']  # , 'cyclic']
             # 'warm_start': [True, False]
         },
         "Lasso CV", {
             'skipPreElim': True,
             'cv'         : 1
         }),
        ("Ridge", linear_model.Ridge(), {
            'alpha': ScistatsNormAround(1, 10, cond=(lambda x: x > 0)),
            'tol'  : ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True)
        }, "Ridge", None),
        ("RidgeCV",
         linear_model.RidgeCV(cv=4, alphas=ScistatsNormAround(1, 100, cond=(lambda x: x > 0)).rvs(size=100)), None,
         "Ridge CV", None),  # TODO improvable
        ("ElasticNet", linear_model.ElasticNet(max_iter=maxiterPos), {
            'alpha'    : ScistatsNormAround(1, 10, cond=(lambda x: x >= 0.1)),
            'l1_ratio' : ScistatsNormBetween(0, 1, clip=True),
            'tol'      : ScistatsNormAround(0, 1e-2, cond=(lambda x: x >= 1e-4), clip=True),
            'selection': ['random']  # , 'cyclic']
            # 'warm_start': [True, False]
        },
         "Elastic Net", None),
        ("ElasticNetCV", linear_model.ElasticNetCV(max_iter=maxiterPos, n_jobs=-1), {
            'l1_ratio' : ScistatsNormBetween(0, 1, clip=True),
            'eps'      : ScistatsNormBetween(1e-4, 1e-2, cond=(lambda x: x > 0)),
            'n_alphas' : ScistatsNormBetween(10, 1000, cond=(lambda x: x >= 10), toint=True),
            'tol'      : ScistatsNormAround(0, 1e-2, cond=(lambda x: x >= 1e-4), clip=True),
            'cv'       : ScistatsNormBetween(2, 4, clip=True, toint=True),
            'selection': ['random']  # , 'cyclic']
            # 'warm_start': [True, False]
        },
         "Elastic Net CV", {
             'skipPreElim': True,
             'cv'         : 1
         }),
        ("BayesianRidge", linear_model.BayesianRidge(n_iter=maxiterPos), {
            'alpha_1'    : ScistatsNormAround(0, 1e-3, cond=(lambda x: 0 < x <= 1e-3)),
            'alpha_2'    : ScistatsNormAround(0, 1e-3, cond=(lambda x: 0 < x <= 1e-3)),
            'lambda_1'   : ScistatsNormAround(0, 1e-3, cond=(lambda x: 0 < x <= 1e-3)),
            'lambda_2'   : ScistatsNormAround(0, 1e-3, cond=(lambda x: 0 < x <= 1e-3)),
            'alpha_init' : ScistatsNormBetween(0, 1, cond=(lambda x: 0 <= x <= 1)),
            'lambda_init': ScistatsNormAround(1, 10, cond=(lambda x: x > 0)),
            'tol'        : ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True)
        },
         "Bayesian Ridge", None),
        ("ARD", linear_model.ARDRegression(), {
            'n_iter'          : ScistatsNormAround(300, 200, cond=(lambda x: x >= 100), toint=True),
            'tol'             : ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True),
            'alpha_1'         : ScistatsNormAround(0, 1e-3, cond=(lambda x: 0 < x <= 1e-3)),
            'alpha_2'         : ScistatsNormAround(0, 1e-3, cond=(lambda x: 0 < x <= 1e-3)),
            'lambda_1'        : ScistatsNormAround(0, 1e-3, cond=(lambda x: 0 < x <= 1e-3)),
            'lambda_2'        : ScistatsNormAround(0, 1e-3, cond=(lambda x: 0 < x <= 1e-3)),
            'threshold_lambda': ScistatsNormBetween(5000, 20000, clip=True)
        },
         "Automatic Relevance Determination Regression", None),
        ("SGD", linear_model.SGDRegressor(max_iter=maxiterPos), {
            'loss'         : ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'penalty'      : ['l2', 'l1', 'elasticnet'],
            'alpha'        : ScistatsNormAround(0, 1e-1, cond=(lambda x: 0 < x <= 1e-1)),
            'l1_ratio'     : ScistatsNormBetween(0, 1, cond=(lambda x: 0 <= x <= 1)),  # only for penalty=elasticnet
            'epsilon'      : ScistatsNormBetween(1e-3, 10, cond=(lambda x: x > 0)),
            # for loss=huber, epsilon_insensitive, squared_epsilon_insensitive
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'eta0'         : ScistatsNormBetween(1e-3, 1e-1, cond=(lambda x: x > 0)),
            'tol'          : ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True),
            # for learning_rate=constant, invscaling, adaptive
            'power_t'      : ScistatsNormBetween(0, 1, cond=(lambda x: 0 < x < 1))  # for learning_rate=invscaling
        },
         "Stochastic Gradient Descent", None),
        ("PA", linear_model.PassiveAggressiveRegressor(max_iter=maxiterPos), {
            'C'      : ScistatsNormAround(0, 1e3, cond=(lambda x: x > 0)),
            'tol'    : ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3), clip=True),
            'loss'   : ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            'epsilon': ScistatsNormAround(0.1, 10, cond=(lambda x: x >= 1e-2))
            # 'warm_start': [True, False]
        },
         "Passive Aggressive Regressor", None),
        ("Huber", linear_model.HuberRegressor(max_iter=maxiterPos), {
            'epsilon': ScistatsNormAround(1, 10, cond=(lambda x: x > 1)),
            'alpha'  : ScistatsNormAround(0, 1e-1, cond=(lambda x: 0 < x <= 1e-1)),
            'tol'    : ScistatsNormAround(0, 1e-3, cond=(lambda x: x >= 1e-5), clip=True)
            # 'warm_start': [True, False]
        },
         "Huber Regressor", None),
        ("TheilSen", linear_model.TheilSenRegressor(n_jobs=-1, max_iter=maxiterPos,
                                                    tol=ScistatsNormAround(0, 1e-1, cond=(lambda x: x >= 1e-3),
                                                                           clip=True).rvs()),
         None, "Theil Sen", None)
    ]
    if restrict is not None:
        models = [x for x in models if x[0] in restrict]
    if randomOrder:
        random.shuffle(models)
    return models


def iround(num):
    return int(round(num, 0))


def sanity_check(pFile, X_train, y_train, X_test, y_test, X_unknown, y_unknown, X_full, y_full, bonusPickleInfo=None, loaded=None):
    if loaded is None:
        loaded = pickle.load(open(pFile, 'br'))
    try:
        longname, regr, test_confidence, deg, full_confidence, unknown_confidence, train_confidence, bonusPickleInfo = loaded
    except:
        longname, regr, test_confidence, deg, full_conf = loaded
    train_confidence = regr.score(X_train, y_train)
    test_confidence = regr.score(X_test, y_test)
    full_confidence = regr.score(X_full, y_full)
    unknown_confidence = regr.score(X_unknown, y_unknown) if len(X_unknown) > 0 else None
    toDump = PickleOut(longname, regr, test_confidence, deg, full_confidence, unknown_confidence, train_confidence, bonusPickleInfo)
    with open(pFile, "bw") as f:
        pickle.dump(toDump, f)
    return toDump


def fit_models(X_train, y_train, X_test, y_test, X_unknown, y_unknown, X_full, y_full, polyDeg, models=None, picklePrefix='', randomOrder=False, notRecompute=True, maxiter=-1,
               onlyImprove=False, modelsToProduce=None, showDone=False, cvPostfix=None, modelsPath=None, sanityCheck=False, saveBest=False, bonusPickleInfo=None, progressBar=None):
    def custom_scoring(est, X, y):
        portion = len(X) / len(X_train)
        portion = math.sqrt(portion)
        if portion < 0.05:
            portion = 0.05
        if portion > 0.8:
            portion = 0.8
        #
        test_portion = iround(portion * len(X_test))
        if test_portion > len(X_test):
            test_portion = len(X_test)
        if test_portion < 2:
            test_portion = 2
        full_portion = iround(portion * len(X_full))
        if full_portion > len(X_full):
            full_portion = len(X_full)
        if full_portion < 2:
            full_portion = 2
        test_samples = random.sample(range(len(X_test)), test_portion)
        full_samples = random.sample(range(len(X_full)), full_portion)
        train_confidence = est.score(X, y)
        test_confidence = est.score([t for i, t in enumerate(X_test) if i in test_samples], [t for i, t in enumerate(y_test) if i in test_samples])
        full_confidence = est.score([t for i, t in enumerate(X_full) if i in full_samples], [t for i, t in enumerate(y_full) if i in full_samples])
        newGeo = jamGeomean([test_confidence, full_confidence, train_confidence])
        return newGeo

    #
    if modelsPath is None:
        modelsPath = "./models"
    if cvPostfix is None:
        cvPostfix = ""
    if modelsToProduce is None:
        modelsToProduce = get_model_names()
    if models is None:
        models = get_models(randomOrder=randomOrder, maxiter=maxiter, restrict=modelsToProduce)
    # printInfo(f"Processing models: {[x[3] for x in models]}")
    trained = []
    for model in models:
        modelName, regr, params, longname, halvingParams = model
        if params is not None:
            cv = ScistatsNormBetween(2, 4, cond=(lambda x: 2 <= x <= 5), toint=True).rvs()
            if halvingParams is not None:
                if 'cv' in halvingParams.keys():
                    cv = halvingParams['cv']
                    if cv is None or cv <= 1:
                        cv = 2
            baseRes = ScistatsNormBetween(8, 20, cond=(lambda x: x >= 2 * cv + 1), toint=True).rvs()
            numTurns = ScistatsNormBetween(cv / 2, 4 * cv, cond=(lambda x: 2 <= x <= 20), toint=True).rvs()
            prelimRounds = ScistatsNormBetween(0, cv / 2, cond=(lambda x: 0 <= x <= 3), toint=True).rvs()
            if halvingParams is not None:
                if 'skipPreElim' in halvingParams.keys() and halvingParams['skipPreElim']:
                    prelimRounds = 0
            if numTurns >= 15:
                prelimRounds = 0
            lastRoundResPercent = ScistatsNormBetween(0.8, 1.0, cond=(lambda x: 0.75 <= x <= 1.0)).rvs()
            lastRoundRes = iround(len(X_train) * lastRoundResPercent)
            fact = (lastRoundRes / baseRes) ** (1 / numTurns)
            lastRoundNumCand = ScistatsNormBetween(1, 10, cond=(lambda x: x >= 1), toint=True).rvs()
            numCand = iround(lastRoundNumCand * (fact ** (numTurns + prelimRounds - 1)))
            searchParams = {
                'estimator'             : regr,
                'param_distributions'   : params,
                'n_jobs'                : -1,
                'factor'                : fact,
                'n_candidates'          : numCand,
                'min_resources'         : baseRes,
                'aggressive_elimination': True,
                'refit'                 : True,
                # 'return_train_score'    : True,
                'scoring'               : custom_scoring,
                'error_score'           : 0,
                'cv'                    : cv,
                'verbose'               : 1
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
                    toDump = PickleOut(longname, regr.best_estimator_, test_confidence, polyDeg, full_confidence, unknown_confidence, train_confidence, bonusPickleInfo)
                else:
                    # toDump = (longname, regr, test_confidence, polyDeg, full_confidence)
                    toDump = PickleOut(longname, regr, test_confidence, polyDeg, full_confidence, unknown_confidence, train_confidence, bonusPickleInfo)
                if onlyImprove:
                    printInfo(f"Trying to improve {fullname}", 1)
                    if not pFile.is_file():
                        printWarn(f"There was no pickle found for {fullname} (at {pickleName}.pickle) to improve", 1)
                        pickle.dump(toDump, open(pFile, 'bw'))
                        trained.append(toDump)
                    else:
                        loaded = pickle.load(open(pFile, 'br'))
                        if sanityCheck:
                            loaded = sanity_check(pFile, X_train, y_train, X_test, y_test, X_unknown, y_unknown, X_full, y_full, bonusPickleInfo, loaded)
                        newGeo = jamGeomean([test_confidence, full_confidence, train_confidence])
                        loadedGeo = jamGeomean([loaded.test_confidence, loaded.full_confidence, loaded.train_confidence])
                        table = tabulate([["old:", loaded.test_confidence, loaded.full_confidence, loaded.train_confidence, loadedGeo],
                                          ["new:", test_confidence, full_confidence, train_confidence, newGeo]],
                                         headers=["", "test_conf", "full_conf", "train_conf", "geo_mean"])
                        if newGeo >= loadedGeo:
                            pickle.dump(toDump, open(pFile, 'bw'))
                            printInfo(
                                f"Improved (or matched) {fullname} (over {pickleName}.pickle): (+{newGeo - loadedGeo})",
                                1)
                            trained.append(toDump)
                        else:
                            printInfo(
                                f"Did not improve {fullname} (over {pickleName}.pickle): ({newGeo - loadedGeo})",
                                1)
                            trained.append(loaded)
                        print("\n".join(["\t" + row for row in table.split("\n")]))
                else:
                    pickle.dump(toDump, open(pFile, 'bw'))
                    printInfo(f"Saved {fullname} (to {pickleName}.pickle)", 1)
                    trained.append(toDump)
                # spinner.succeed(f"Finished fitting {fullName}")
                printSucc(f"Finished fitting {fullname}", 1)
            except Exception as e:
                # spinner.fail("failed")
                printError(f"Failed to fit {fullname} with error:", 1)
                print(e)
                exit(1)
        else:
            if showDone:
                printInfo(f"{fullname}:")
                if not pFile.is_file():
                    printInfo("Found no pickle", 1)
                    trained.append(PickleOut(longname, None, None, polyDeg, None, None, None, None))
                else:
                    printInfo("Found pickle", 1)
                    loaded = pickle.load(open(pFile, 'br'))
                    if sanityCheck:
                        loaded = sanity_check(pFile, X_train, y_train, X_test, y_test, X_unknown, y_unknown, X_full, y_full, bonusPickleInfo, loaded)
                    trained.append(loaded)
                    printSucc("Sanity checked", 1)
            else:
                printInfo(f"{fullname}:")
                loaded = pickle.load(open(pFile, 'br'))
                if sanityCheck:
                    loaded = sanity_check(pFile, X_train, y_train, X_test, y_test, X_unknown, y_unknown, X_full, y_full, bonusPickleInfo, loaded)
                trained.append(loaded)
                # spinner.succeed(f"Found pickle for {fullName}")
                printSucc(f"Found pickle for {fullname} at {pickleName}.pickle", 1)
        #
        if progressBar is not None:
            progressBar()
    #
    if saveBest:
        bestModelPath = Path(f"{modelsPath}/bestModel.pickle")
        if bestModelPath.exists():
            bestModel = list(pickle.load(open(bestModelPath, 'br')))
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
        pickle.dump((name, loadedRegr, conf, deg), open(f"{modelsPath}/bestModel.pickle", 'bw'))
    #
    return trained


if __name__ == '__main__':
    main()
