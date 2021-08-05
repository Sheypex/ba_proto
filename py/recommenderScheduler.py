import pickle
import re
from dataclasses import dataclass
from typing import List

import networkx as NX
from db_actions import db_actions
import pandas as pds
import matplotlib.pyplot as plt
import itertools
import math
import re
import statistics
from functools import reduce
from pprint import pprint
from log_symbols import LogSymbols
from sklearn import linear_model, metrics, preprocessing, model_selection, svm
from tabulate import tabulate
from pathlib import Path
import pickle
from halo import Halo
import random
import argparse
import sys
from db_actions import db_actions
import numpy as np
from my_yaml import yaml_load, yaml_dump


class Process:
    name: str
    start: float
    end: float
    dur: float
    running: bool
    runningOnInst: int

    def __init__(self, name, start=None, dur=None, end=None):
        self.name = name
        self.start = start
        self.dur = dur
        self.end = end
        self.running = False

    def startProcess(self, time, dur=None, runningOnInst=None):
        self.start = time
        self.running = True
        self.runningOnInst = runningOnInst
        if dur is not None:
            self.dur = dur
            self.end = time + dur
        else:
            assert self.dur is not None
            self.end = time + self.dur

    def isDone(self, time):
        return time >= self.end

    def __eq__(self, other):
        if isinstance(other, Process):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            raise TypeError(f"Cannot compare Process to {other} since it's not of type Process or str.")

    def __repr__(self):
        if self.start is not None and self.end is not None:
            return f"Process({self.name}, from {self.start} to {self.end})"
        else:
            return f"Process({self.name})"

    def __str__(self):
        return self.__repr__()


def main():
    regModel: linear_model.LinearRegression = pickle.load(open('models/linModel.SVR-rbf.pickle', 'br'))[1]
    with db_actions.connect() as conn:
        wfNames = list(pds.read_sql("select distinct \"wfName\" from \"averageRuntimesPredictionBase\"",
                                    conn).wfName.values)  # ['nfcore/chipseq:1.2.2', 'nfcore/eager:2.3.5', 'nfcore/methylseq:1.6.1', 'nfcore/sarek:2.7.1', 'nfcore/viralrecon:2.1']
        predBase = pds.read_sql(
            "SELECT * FROM \"averageRuntimesPredictionBase\" WHERE realtime > 1000",
            conn)
        taskRuntimes = pds.read_sql(
            "SELECT * FROM \"taskRuntimeAverages\" WHERE realtime > 1000",
            conn)
        nodeBenchmarks = pds.read_sql(
            "SELECT * FROM \"nodeBenchmarkTransposedRankings\"",
            conn)
    scale = preprocessing.StandardScaler().fit(
        predBase.drop(['wfName', 'taskName', 'nodeConfig', 'realtime', 'rank'], axis=1))
    poly = preprocessing.PolynomialFeatures(degree=1, interaction_only=True, include_bias=False).fit(
        predBase.drop(['wfName', 'taskName', 'nodeConfig', 'realtime', 'rank'], axis=1))  # TODO: depends on model used!
    scale2 = preprocessing.StandardScaler().fit(
        poly.transform(scale.transform(predBase.drop(['wfName', 'taskName', 'nodeConfig', 'realtime', 'rank'],
                                                     axis=1))))  # TODO: pickle these once on AWS
    # test
    # t = predBase.iloc[[0]]
    # # print(t)
    # machine = t['nodeConfig'].values[0]
    # task = t['taskName'].values[0]
    # wf = t['wfName'].values[0]
    # rank = t['rank'].values[0]
    # print(machine, wf, rank)
    # t = t.drop(['wfName', 'taskName', 'nodeConfig', 'realtime', 'rank'], axis=1)
    # t = scale.transform(t)
    # t = poly.transform(t)
    # t = scale2.transform(t)
    # # print(t)
    # r = regModel.predict(t)
    # print(r, rank)
    # print(regModel.score(scale2.transform(poly.transform(
    #     scale.transform(predBase.drop(['wfName', 'taskName', 'nodeConfig', 'realtime', 'rank'], axis=1)))),
    #     predBase[['rank']]))
    # valid instances: [165 , 193] \ {174,177}  # c5.- (174) and c5a.large (177) are out
    # TODO: get these as CLArgs
    instances = [191, 176, 187, 193, 166, 172, 171, 192, 181,
                 175]  # https://www.random.org/integers/?num=10&min=165&max=193&col=10&base=10&format=html&rnd=new
    # filter for instances we have available (probably should rather do this in the sql query(ies) above)
    predBase.query("nodeConfig in @instances", inplace=True)
    taskRuntimes.query("nodeConfig in @instances", inplace=True)
    nodeBenchmarks.query("nodeConfig in @instances", inplace=True)
    #
    for wfName in wfNames:
        wfShortName = re.compile("nfcore/(\w+):.*").match(wfName).group(1)
        print(f"{'-=' * 5}{'  ' + wfShortName:<{len(wfShortName) + 4}}{'=-' * 5}")
        #
        time: float = 0.0
        runningTasks: List[Process] = list()
        availableInstances: List[int] = instances
        # schedule = []  # prob unnecessary
        #
        workflowGraph: NX.DiGraph = NX.drawing.nx_agraph.read_dot(f"dot/{wfShortName}_simple1000.dot")
        nodeToProcess = {n: Process(n) for n in workflowGraph.nodes()}
        processes = list(nodeToProcess.values())
        # schedule = []
        while len(list(workflowGraph.nodes())) > 0:
            ready: List[int] = [t[0] for t in list(workflowGraph.in_degree()) if
                                t[1] == 0]  # all tasks that are ready to run
            ready: List[int] = [r for r in ready if
                                r not in [p.name for p in runningTasks]]  # all ready tasks that aren't running yet
            if len(ready) > 0:
                readyProcs: List[Process] = [nodeToProcess[r] for r in ready]
                runningTasks += readyProcs
                for p in readyProcs:
                    # find best instance
                    predictedRanks = {}
                    for inst in availableInstances:
                        predRow = predBase.query(
                            "(nodeConfig == @inst) and (taskName == @p.name) and (wfName == @wfName)")
                        r = regModel.predict(scale2.transform(poly.transform(scale.transform(
                            predRow.drop(['wfName', 'taskName', 'nodeConfig', 'realtime', 'rank'], axis=1)))))
                        predictedRanks[inst] = r[0]
                    bestInst: int = [n for n, r in predictedRanks.items() if
                                     r == min([pr for pr in predictedRanks.values()])][
                        0]  # TODO: this arbitrarily chooses the first instance among all that have the highest rank # TODO: should the predicted ranks be rounded??
                    # find corresponding duration on that best instance
                    taskDurOnBestInst = list(
                        predBase.query("(nodeConfig == @bestInst) and (taskName == @p.name) and (wfName == @wfName)")[
                            'realtime'].values)[0]
                    # make instance unavailable
                    availableInstances.remove(bestInst)
                    # start process with correct duration
                    p.startProcess(time, dur=taskDurOnBestInst, runningOnInst=bestInst)  # TODO
            else:
                earliest = min([p.end for p in runningTasks])
                finishedProcs = [a for a in runningTasks if a.end == earliest]
                time = earliest
                for p in finishedProcs:
                    runningTasks.remove(p)
                    availableInstances.append(p.runningOnInst)  # make instance the task ran on available again
                    workflowGraph.remove_node(p.name)
        print(time)
        # ready = [t[0] for t in list(wfG.in_degree()) if t[1] == 0]


if __name__ == "__main__":
    main()
