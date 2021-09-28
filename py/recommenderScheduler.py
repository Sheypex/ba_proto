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


class Process:
    name: str
    start: float
    end: float
    dur: float
    running: bool
    runningOnInst: "Instance"

    def __init__(self, name, start=None, dur=None, end=None):
        self.name = name
        self.start = start
        self.dur = dur
        self.end = end
        self.running = False
        self.runningOnInst = None

    def startProcess(self, time, dur=None, runningOnInst: "Instance" = None):
        self.start = time
        self.running = True
        self.runningOnInst = runningOnInst
        runningOnInst.executeProcess(self)
        if dur is not None:
            self.dur = dur
            self.end = time + dur
        else:
            assert self.dur is not None
            self.end = time + self.dur
        # CL.print(f"Started {self!r}")

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
        ret = f"Process({self.name}"
        if self.start is not None and self.end is not None:
            ret += f", from {self.start} to {self.end}"
        if self.runningOnInst is not None:
            ret += f", on {self.runningOnInst}"
        ret += ")"
        return ret

    def __str__(self):
        return self.__repr__()


class Instance:
    id: int
    vcpus: int
    ramgb: int
    hasGpu: bool
    diskType: str
    name: str
    storageType: str
    executingTasks: List[Process]

    def __init__(
        self, id, name=None, vcpus=None, ramgb=None, hasGpu=None, diskType=None, storageType=None,
    ):
        self.id = id
        self.name = name
        self.vcpus = vcpus
        self.ramgb = ramgb
        self.hasGpu = hasGpu
        self.diskType = diskType
        self.storageType = storageType
        self.executingTasks = list()

    # @property
    # def occupiedResources(self):
    #     occ = dict(
    #             vcpus=sum()
    #             )
    #     return occ

    @property
    def hasRoomForMoreProc(self):
        return 4 * len(self.executingTasks) < self.vcpus

    @property
    def remainingCapacity(self):
        if not self.hasRoomForMoreProc:
            return 0
        return math.ceil((self.vcpus - 4 * len(self.executingTasks)) / 4)

    @property
    def numRunning(self):
        return len(self.executingTasks)

    def executeProcess(self, proc):
        if not self.hasRoomForMoreProc:
            raise Exception("This instance cannot execute another task.")
        self.executingTasks.append(proc)

    def finishProcess(self, proc):
        self.executingTasks.remove(proc)

    def __eq__(self, other):
        if isinstance(other, Instance):
            return self.id == other.id
        elif isinstance(other, int):
            return self.id == other
        else:
            raise TypeError(f"Instance is only comparable to Instance and int. other was {other.__class__}.")

    def __repr__(self):
        ret = f"Instance({self.id}-'{self.name}', {self.vcpus})"
        return ret

    def __str__(self):
        return self.__repr__()


class Scheduler(ABC):
    def __init__(
        self, workflowGraph: NX.DiGraph, instances: List[Instance], rankLookup, realtimeLookup,
    ):
        self.time = 0.0
        self.runningTasks: List[Process] = list()
        self.availableInstances: List[Instance] = instances.copy()
        self.trace: List[List[Process]] = list()
        #
        self.nodeToProcess = {n: Process(n) for n in workflowGraph.nodes()}
        self.processes = list(self.nodeToProcess.values())
        #
        self.workflowGraph = workflowGraph
        self.rankLookup = rankLookup
        self.realtimeLookup = realtimeLookup

    @abstractmethod
    def dispatch(self, readyProcs: List[Process]):
        ...

    def schedule(self):
        while len(list(self.workflowGraph.nodes())) > 0:
            ready: List[int] = [
                t[0] for t in list(self.workflowGraph.in_degree()) if t[1] == 0
            ]  # all tasks that are ready to run
            ready: List[int] = [
                r for r in ready if r not in [p.name for p in self.runningTasks]
            ]  # all ready tasks that aren't running yet
            if len(ready) > 0 and sum(i.hasRoomForMoreProc for i in self.availableInstances) > 0:
                readyProcs: List[Process] = [self.nodeToProcess[r] for r in ready]
                self.dispatch(readyProcs)
            else:
                earliest = min(p.end for p in self.runningTasks)
                finishedProcs = [a for a in self.runningTasks if a.end == earliest]
                self.time = earliest
                for p in finishedProcs:
                    self.runningTasks.remove(p)
                    p.runningOnInst.finishProcess(p)
                    self.workflowGraph.remove_node(p.name)
            if len(self.runningTasks) > 0:
                self.trace.append(self.runningTasks.copy())
        return self.time, self.trace


class RecommenderSchedulerV1(Scheduler):
    def dispatch(self, readyProcs: List[Process]):
        for p in readyProcs:
            avInst = [i for i in self.availableInstances if i.hasRoomForMoreProc]
            # find best instance
            if len(avInst) > 0:
                self.runningTasks.append(p)
                predictedRanks = list()
                for inst in avInst:
                    predictedRanks.append((inst, self.rankLookup.at[inst.id, p.name]))
                bestRank = min(pr for _, pr in predictedRanks)
                bestInst: Instance = [n for n, r in predictedRanks if r == bestRank][
                    0
                ]  # TODO: this arbitrarily chooses the first instance among all that have the highest rank # TODO: should the predicted ranks be rounded??
                # find corresponding duration on that best instance
                taskDurOnBestInst = self.realtimeLookup.at[bestInst.id, p.name]
                # make instance unavailable
                # self.availableInstances.remove(bestInst)
                # start process with correct duration
                p.startProcess(self.time, dur=taskDurOnBestInst, runningOnInst=bestInst)
            else:
                break


def recommenderSchedulerV1(workflowGraph: NX.DiGraph, instances: List[int], rankLookup, realtimeLookup):
    time: float = 0.0
    runningTasks: List[Process] = list()
    availableInstances: List[int] = instances
    trace: List[List[Process]] = list()
    #
    nodeToProcess = {n: Process(n) for n in workflowGraph.nodes()}
    processes = list(nodeToProcess.values())
    # schedule = []
    while len(list(workflowGraph.nodes())) > 0:
        ready: List[int] = [
            t[0] for t in list(workflowGraph.in_degree()) if t[1] == 0
        ]  # all tasks that are ready to run
        ready: List[int] = [
            r for r in ready if r not in [p.name for p in runningTasks]
        ]  # all ready tasks that aren't running yet
        if len(ready) > 0 and len(availableInstances) > 0:
            readyProcs: List[Process] = [nodeToProcess[r] for r in ready]
            for p in readyProcs:
                # find best instance
                if len(availableInstances) > 0:
                    runningTasks.append(p)
                    predictedRanks = {}
                    for inst in list(set(availableInstances)):
                        predictedRanks[inst] = rankLookup.at[inst, p.name]
                    bestInst: int = [
                        n for n, r in predictedRanks.items() if r == min([pr for pr in predictedRanks.values()])
                    ][
                        0
                    ]  # TODO: this arbitrarily chooses the first instance among all that have the highest rank # TODO: should the predicted ranks be rounded??
                    # find corresponding duration on that best instance
                    taskDurOnBestInst = realtimeLookup.at[bestInst, p.name]
                    # make instance unavailable
                    availableInstances.remove(bestInst)
                    # start process with correct duration
                    p.startProcess(time, dur=taskDurOnBestInst, runningOnInst=bestInst)  # TODO
                else:
                    break
            trace.append(runningTasks.copy())
        else:
            earliest = min([p.end for p in runningTasks])
            finishedProcs = [a for a in runningTasks if a.end == earliest]
            time = earliest
            for p in finishedProcs:
                runningTasks.remove(p)
                availableInstances.append(p.runningOnInst)  # make instance the task ran on available again
                workflowGraph.remove_node(p.name)
            trace.append(runningTasks.copy())
    trace = [
        t for t in trace if len(t) > 0
    ]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    return time, trace


class RecommenderSchedulerV1H1(RecommenderSchedulerV1):
    def dispatch(self, readyProcs: List[Process]):
        super().dispatch(sorted(readyProcs, key=lambda x: -self.workflowGraph.out_degree(x.name)))


def recommenderSchedulerV1H1(workflowGraph: NX.DiGraph, instances: List[int], rankLookup, realtimeLookup):
    # Heuristic #1: prioritize ready tasks with many direct descendants
    time: float = 0.0
    runningTasks: List[Process] = list()
    availableInstances: List[int] = instances
    trace: List[List[Process]] = list()
    #
    nodeToProcess = {n: Process(n) for n in workflowGraph.nodes()}
    processes = list(nodeToProcess.values())
    # schedule = []
    while len(list(workflowGraph.nodes())) > 0:
        ready: List[int] = [
            t[0] for t in list(workflowGraph.in_degree()) if t[1] == 0
        ]  # all tasks that are ready to run
        ready: List[int] = [
            r for r in ready if r not in [p.name for p in runningTasks]
        ]  # all ready tasks that aren't running yet
        if len(ready) > 0 and len(availableInstances) > 0:
            ready.sort(key=lambda x: -workflowGraph.out_degree(x))  # H1
            readyProcs: List[Process] = [nodeToProcess[r] for r in ready]
            for p in readyProcs:
                # find best instance
                if len(availableInstances) > 0:
                    runningTasks.append(p)
                    predictedRanks = {}
                    for inst in list(set(availableInstances)):
                        predictedRanks[inst] = rankLookup.at[inst, p.name]
                    bestInst: int = [
                        n for n, r in predictedRanks.items() if r == min([pr for pr in predictedRanks.values()])
                    ][
                        0
                    ]  # TODO: this arbitrarily chooses the first instance among all that have the highest rank # TODO: should the predicted ranks be rounded??
                    # find corresponding duration on that best instance
                    taskDurOnBestInst = realtimeLookup.at[bestInst, p.name]
                    # make instance unavailable
                    availableInstances.remove(bestInst)
                    # start process with correct duration
                    p.startProcess(time, dur=taskDurOnBestInst, runningOnInst=bestInst)  # TODO
                else:
                    break
            trace.append(runningTasks.copy())
        else:
            earliest = min([p.end for p in runningTasks])
            finishedProcs = [a for a in runningTasks if a.end == earliest]
            time = earliest
            for p in finishedProcs:
                runningTasks.remove(p)
                availableInstances.append(p.runningOnInst)  # make instance the task ran on available again
                workflowGraph.remove_node(p.name)
            trace.append(runningTasks.copy())
    trace = [
        t for t in trace if len(t) > 0
    ]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    return time, trace


class RecommenderSchedulerV1H2(RecommenderSchedulerV1):
    def dispatch(self, readyProcs: List[Process]):
        super().dispatch(
            sorted(readyProcs, key=lambda x: -(len(NX.algorithms.dag.descendants(self.workflowGraph, x.name))))
        )


def recommenderSchedulerV1H2(workflowGraph: NX.DiGraph, instances: List[int], rankLookup, realtimeLookup):
    # Heuristic #2: prioritize ready tasks with many total descendants
    time: float = 0.0
    runningTasks: List[Process] = list()
    availableInstances: List[int] = instances
    trace: List[List[Process]] = list()
    #
    nodeToProcess = {n: Process(n) for n in workflowGraph.nodes()}
    processes = list(nodeToProcess.values())
    # schedule = []
    while len(list(workflowGraph.nodes())) > 0:
        ready: List[int] = [
            t[0] for t in list(workflowGraph.in_degree()) if t[1] == 0
        ]  # all tasks that are ready to run
        ready: List[int] = [
            r for r in ready if r not in [p.name for p in runningTasks]
        ]  # all ready tasks that aren't running yet
        if len(ready) > 0 and len(availableInstances) > 0:
            ready.sort(key=lambda x: -(len(NX.algorithms.dag.descendants(workflowGraph, x))))  # H2
            readyProcs: List[Process] = [nodeToProcess[r] for r in ready]
            for p in readyProcs:
                # find best instance
                if len(availableInstances) > 0:
                    runningTasks.append(p)
                    predictedRanks = {}
                    for inst in list(set(availableInstances)):
                        predictedRanks[inst] = rankLookup.at[inst, p.name]
                    bestInst: int = [
                        n for n, r in predictedRanks.items() if r == min([pr for pr in predictedRanks.values()])
                    ][
                        0
                    ]  # TODO: this arbitrarily chooses the first instance among all that have the highest rank # TODO: should the predicted ranks be rounded??
                    # find corresponding duration on that best instance
                    taskDurOnBestInst = realtimeLookup.at[bestInst, p.name]
                    # make instance unavailable
                    availableInstances.remove(bestInst)
                    # start process with correct duration
                    p.startProcess(time, dur=taskDurOnBestInst, runningOnInst=bestInst)  # TODO
                else:
                    break
            trace.append(runningTasks.copy())
        else:
            earliest = min([p.end for p in runningTasks])
            finishedProcs = [a for a in runningTasks if a.end == earliest]
            time = earliest
            for p in finishedProcs:
                runningTasks.remove(p)
                availableInstances.append(p.runningOnInst)  # make instance the task ran on available again
                workflowGraph.remove_node(p.name)
            trace.append(runningTasks.copy())
    trace = [
        t for t in trace if len(t) > 0
    ]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    return time, trace


class RecommenderSchedulerV1H3(RecommenderSchedulerV1):
    def dispatch(self, readyProcs: List[Process]):
        super().dispatch(sorted(readyProcs, key=self.tmpSort))

    @property
    @cache
    def sinks(self):
        return [t[0] for t in list(self.workflowGraph.out_degree()) if t[1] == 0]

    def tmpSort(self, x):
        if x not in self.sinks:
            return -max(len(a) for a in list(NX.algorithms.all_simple_paths(self.workflowGraph, x.name, self.sinks)))
        else:
            return 0


def recommenderSchedulerV1H3(workflowGraph: NX.DiGraph, instances: List[int], rankLookup, realtimeLookup):
    # Heuristic #3: prioritize ready tasks by maximum length of chains of descendants
    time: float = 0.0
    runningTasks: List[Process] = list()
    availableInstances: List[int] = instances.copy()
    trace: List[List[Process]] = list()
    #
    nodeToProcess = {n: Process(n) for n in workflowGraph.nodes()}
    processes = list(nodeToProcess.values())
    #
    sinks = [t[0] for t in list(workflowGraph.out_degree()) if t[1] == 0]

    def tmpSort(x):
        if x not in sinks:
            return -max([len(a) for a in list(NX.algorithms.all_simple_paths(workflowGraph, x, sinks))])
        else:
            return 0

    #
    while len(list(workflowGraph.nodes())) > 0:
        ready: List[int] = [
            t[0] for t in list(workflowGraph.in_degree()) if t[1] == 0
        ]  # all tasks that are ready to run
        ready: List[int] = [
            r for r in ready if r not in [p.name for p in runningTasks]
        ]  # all ready tasks that aren't running yet
        if len(ready) > 0 and len(availableInstances) > 0:
            ready.sort(key=tmpSort)  # H3
            readyProcs: List[Process] = [nodeToProcess[r] for r in ready]
            for p in readyProcs:
                # find best instance
                if len(availableInstances) > 0:
                    runningTasks.append(p)
                    predictedRanks = {}
                    for inst in list(set(availableInstances)):
                        predictedRanks[inst] = rankLookup.at[inst, p.name]
                    bestInst: int = [
                        n for n, r in predictedRanks.items() if r == min([pr for pr in predictedRanks.values()])
                    ][
                        0
                    ]  # TODO: this arbitrarily chooses the first instance among all that have the highest rank # TODO: should the predicted ranks be rounded??
                    # find corresponding duration on that best instance
                    taskDurOnBestInst = realtimeLookup.at[bestInst, p.name]
                    # make instance unavailable
                    availableInstances.remove(bestInst)
                    # start process with correct duration
                    p.startProcess(time, dur=taskDurOnBestInst, runningOnInst=bestInst)  # TODO
                else:
                    break
            trace.append(runningTasks.copy())
        else:
            earliest = min([p.end for p in runningTasks])
            finishedProcs = [a for a in runningTasks if a.end == earliest]
            time = earliest
            for p in finishedProcs:
                runningTasks.remove(p)
                availableInstances.append(p.runningOnInst)  # make instance the task ran on available again
                workflowGraph.remove_node(p.name)
            trace.append(runningTasks.copy())
    # CL.print(time)
    trace = [
        t for t in trace if len(t) > 0
    ]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    # for i, t in enumerate(trace):
    #     CL.print(f"{i}\t", end='')
    #     CL.print(t)
    return time, trace


class RecommenderSchedulerV2(Scheduler):
    @property
    def availableSlots(self):
        return [j for i in self.availableInstances for j in [i] * i.remainingCapacity if i.hasRoomForMoreProc]

    def dispatch(self, readyProcs: List[Process]):
        for p in readyProcs:
            slots = self.availableSlots
            # find best instance
            if len(slots) > 0:
                canFit = min(len(slots), len(readyProcs))
                procComb = itertools.combinations(readyProcs, canFit)
                instComb = itertools.combinations(slots, canFit)
                mappings = itertools.product(procComb, instComb)
                scores = list()
                for m in mappings:
                    scores.append(
                        (zip(m[0], m[1]), sum(self.rankLookup.at[inst.id, p.name] for p, inst in zip(m[0], m[1])))
                    )
                scores.sort(key=lambda x: x[1])
                bestMapping = scores[0]  # TODO: arbitrary selection for multiple best mappings with minimal score
                for y in bestMapping[0]:
                    p, inst = y
                    # find corresponding duration on that best instance
                    taskDurOnBestInst = self.realtimeLookup.at[inst.id, p.name]
                    # make instance unavailable
                    # self.availableInstances.remove(inst)
                    # start process with correct duration
                    p.startProcess(self.time, dur=taskDurOnBestInst, runningOnInst=inst)
                    self.runningTasks.append(p)
            else:
                break


def recommenderSchedulerV2(workflowGraph: NX.DiGraph, instances: List[int], rankLookup, realtimeLookup):
    # Version #2: check all ready tasks for instance rankings before dispatching to get lowest sum of rankings on concurrent tasks
    time: float = 0.0
    runningTasks: List[Process] = list()
    availableInstances: List[int] = instances
    trace: List[List[Process]] = list()
    #
    nodeToProcess = {n: Process(n) for n in workflowGraph.nodes()}
    processes = list(nodeToProcess.values())
    # schedule = []
    while len(list(workflowGraph.nodes())) > 0:
        ready: List[int] = [
            t[0] for t in list(workflowGraph.in_degree()) if t[1] == 0
        ]  # all tasks that are ready to run
        ready: List[int] = [
            r for r in ready if r not in [p.name for p in runningTasks]
        ]  # all ready tasks that aren't running yet
        if len(ready) > 0 and len(availableInstances) > 0:
            readyProcs: List[Process] = [nodeToProcess[r] for r in ready]
            procComb = itertools.combinations(readyProcs, min([len(availableInstances), len(readyProcs)]))
            instComb = itertools.combinations(availableInstances, min([len(availableInstances), len(readyProcs)]))
            mappings = list(itertools.product(list(procComb), list(instComb)))
            mappings = [list(zip(m[0], m[1])) for m in mappings]
            for i, m in enumerate(mappings):
                r = []
                for j, x in enumerate(m):
                    p, inst = x
                    r.append((p, inst, rankLookup.at[inst, p.name]))
                mappings[i] = tuple(r)
            for i, m in enumerate(mappings):
                s = sum([x[2] for x in m])
                mappings[i] = (m, s)
            mappings.sort(key=lambda x: x[1])
            bestMapping = mappings[0]
            for y in bestMapping[0]:
                p, inst, _ = y
                # find corresponding duration on that best instance
                taskDurOnBestInst = realtimeLookup.at[inst, p.name]
                # make instance unavailable
                availableInstances.remove(inst)
                # start process with correct duration
                p.startProcess(time, dur=taskDurOnBestInst, runningOnInst=inst)
                runningTasks.append(p)
        else:
            earliest = min([p.end for p in runningTasks])
            finishedProcs = [a for a in runningTasks if a.end == earliest]
            time = earliest
            for p in finishedProcs:
                runningTasks.remove(p)
                availableInstances.append(p.runningOnInst)  # make instance the task ran on available again
                workflowGraph.remove_node(p.name)
        trace.append(runningTasks.copy())
    trace = [
        t for t in trace if len(t) > 0
    ]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    return time, trace


class RecommenderSchedulerV2H1(RecommenderSchedulerV2):
    def dispatch(self, readyProcs: List[Process]):
        super().dispatch(
            sorted(readyProcs, key=lambda x: -self.workflowGraph.out_degree(x.name))[
                0 : min([len(self.availableSlots), len(readyProcs)])
            ]
        )


def recommenderSchedulerV2H1(workflowGraph: NX.DiGraph, instances: List[int], rankLookup, realtimeLookup):
    # Version #2: check all ready tasks for instance rankings before dispatching to get lowest sum of rankings on concurrent tasks
    # Heuristic #1: prioritize ready tasks with many direct descendants
    time: float = 0.0
    runningTasks: List[Process] = list()
    availableInstances: List[int] = instances
    trace: List[List[Process]] = list()
    #
    nodeToProcess = {n: Process(n) for n in workflowGraph.nodes()}
    processes = list(nodeToProcess.values())
    # schedule = []
    while len(list(workflowGraph.nodes())) > 0:
        ready: List[int] = [
            t[0] for t in list(workflowGraph.in_degree()) if t[1] == 0
        ]  # all tasks that are ready to run
        ready: List[int] = [
            r for r in ready if r not in [p.name for p in runningTasks]
        ]  # all ready tasks that aren't running yet
        if len(ready) > 0 and len(availableInstances) > 0:
            ready.sort(key=lambda x: -workflowGraph.out_degree(x))  # H1
            readyProcs: List[Process] = [nodeToProcess[r] for r in ready]
            procComb = [readyProcs[0 : min([len(availableInstances), len(readyProcs)])]]
            instComb = itertools.combinations(availableInstances, min([len(availableInstances), len(readyProcs)]))
            mappings = list(itertools.product(list(procComb), list(instComb)))
            mappings = [list(zip(m[0], m[1])) for m in mappings]
            for i, m in enumerate(mappings):
                r = []
                for j, x in enumerate(m):
                    p, inst = x
                    r.append((p, inst, rankLookup.at[inst, p.name]))
                mappings[i] = tuple(r)
            for i, m in enumerate(mappings):
                s = sum([x[2] for x in m])
                mappings[i] = (m, s)
            mappings.sort(key=lambda x: x[1])
            bestMapping = mappings[0]
            for y in bestMapping[0]:
                p, inst, _ = y
                # find corresponding duration on that best instance
                taskDurOnBestInst = realtimeLookup.at[inst, p.name]
                # make instance unavailable
                availableInstances.remove(inst)
                # start process with correct duration
                p.startProcess(time, dur=taskDurOnBestInst, runningOnInst=inst)
                runningTasks.append(p)
        else:
            earliest = min([p.end for p in runningTasks])
            finishedProcs = [a for a in runningTasks if a.end == earliest]
            time = earliest
            for p in finishedProcs:
                runningTasks.remove(p)
                availableInstances.append(p.runningOnInst)  # make instance the task ran on available again
                workflowGraph.remove_node(p.name)
        trace.append(runningTasks.copy())
    trace = [
        t for t in trace if len(t) > 0
    ]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    return time, trace


class RecommenderSchedulerV2H2(RecommenderSchedulerV2):
    def dispatch(self, readyProcs: List[Process]):
        super().dispatch(
            sorted(readyProcs, key=lambda x: -(len(NX.algorithms.dag.descendants(self.workflowGraph, x.name))))[
                0 : min([len(self.availableSlots), len(readyProcs)])
            ]
        )


def recommenderSchedulerV2H2(workflowGraph: NX.DiGraph, instances: List[int], rankLookup, realtimeLookup):
    # Version #2: check all ready tasks for instance rankings before dispatching to get lowest sum of rankings on concurrent tasks
    # Heuristic #2: prioritize ready tasks with many total descendants
    time: float = 0.0
    runningTasks: List[Process] = list()
    availableInstances: List[int] = instances
    trace: List[List[Process]] = list()
    #
    nodeToProcess = {n: Process(n) for n in workflowGraph.nodes()}
    processes = list(nodeToProcess.values())
    # schedule = []
    while len(list(workflowGraph.nodes())) > 0:
        ready: List[int] = [
            t[0] for t in list(workflowGraph.in_degree()) if t[1] == 0
        ]  # all tasks that are ready to run
        ready: List[int] = [
            r for r in ready if r not in [p.name for p in runningTasks]
        ]  # all ready tasks that aren't running yet
        if len(ready) > 0 and len(availableInstances) > 0:
            ready.sort(key=lambda x: -(len(NX.algorithms.dag.descendants(workflowGraph, x))))  # H2
            readyProcs: List[Process] = [nodeToProcess[r] for r in ready]
            procComb = [readyProcs[0 : min([len(availableInstances), len(readyProcs)])]]
            instComb = itertools.combinations(availableInstances, min([len(availableInstances), len(readyProcs)]))
            mappings = list(itertools.product(list(procComb), list(instComb)))
            mappings = [list(zip(m[0], m[1])) for m in mappings]
            for i, m in enumerate(mappings):
                r = []
                for j, x in enumerate(m):
                    p, inst = x
                    r.append((p, inst, rankLookup.at[inst, p.name]))
                mappings[i] = tuple(r)
            for i, m in enumerate(mappings):
                s = sum([x[2] for x in m])
                mappings[i] = (m, s)
            mappings.sort(key=lambda x: x[1])
            bestMapping = mappings[0]
            for y in bestMapping[0]:
                p, inst, _ = y
                # find corresponding duration on that best instance
                taskDurOnBestInst = realtimeLookup.at[inst, p.name]
                # make instance unavailable
                availableInstances.remove(inst)
                # start process with correct duration
                p.startProcess(time, dur=taskDurOnBestInst, runningOnInst=inst)
                runningTasks.append(p)
        else:
            earliest = min([p.end for p in runningTasks])
            finishedProcs = [a for a in runningTasks if a.end == earliest]
            time = earliest
            for p in finishedProcs:
                runningTasks.remove(p)
                availableInstances.append(p.runningOnInst)  # make instance the task ran on available again
                workflowGraph.remove_node(p.name)
        trace.append(runningTasks.copy())
    trace = [
        t for t in trace if len(t) > 0
    ]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    return time, trace


class RecommenderSchedulerV2H3(RecommenderSchedulerV2):
    def dispatch(self, readyProcs: List[Process]):
        super().dispatch(sorted(readyProcs, key=self.tmpSort)[0 : min([len(self.availableSlots), len(readyProcs)])])

    @property
    @cache
    def sinks(self):
        return [t[0] for t in list(self.workflowGraph.out_degree()) if t[1] == 0]

    def tmpSort(self, x):
        if x not in self.sinks:
            return -max(len(a) for a in list(NX.algorithms.all_simple_paths(self.workflowGraph, x.name, self.sinks)))
        else:
            return 0


def recommenderSchedulerV2H3(workflowGraph: NX.DiGraph, instances: List[int], rankLookup, realtimeLookup):
    # Version #2: check all ready tasks for instance rankings before dispatching to get lowest sum of rankings on concurrent tasks
    # Heuristic #3: prioritize ready tasks by maximum length of chains of descendants
    time: float = 0.0
    runningTasks: List[Process] = list()
    availableInstances: List[int] = instances
    trace: List[List[Process]] = list()
    #
    nodeToProcess = {n: Process(n) for n in workflowGraph.nodes()}
    processes = list(nodeToProcess.values())
    #
    sinks = [t[0] for t in list(workflowGraph.out_degree()) if t[1] == 0]

    def tmpSort(x):
        if x not in sinks:
            return -max([len(a) for a in list(NX.algorithms.all_simple_paths(workflowGraph, x, sinks))])
        else:
            return 0

    #
    while len(list(workflowGraph.nodes())) > 0:
        ready: List[int] = [
            t[0] for t in list(workflowGraph.in_degree()) if t[1] == 0
        ]  # all tasks that are ready to run
        ready: List[int] = [
            r for r in ready if r not in [p.name for p in runningTasks]
        ]  # all ready tasks that aren't running yet
        if len(ready) > 0 and len(availableInstances) > 0:
            ready.sort(key=tmpSort)  # H3
            readyProcs: List[Process] = [nodeToProcess[r] for r in ready]
            procComb = [readyProcs[0 : min([len(availableInstances), len(readyProcs)])]]
            instComb = itertools.combinations(availableInstances, min([len(availableInstances), len(readyProcs)]))
            mappings = list(itertools.product(list(procComb), list(instComb)))
            mappings = [list(zip(m[0], m[1])) for m in mappings]
            for i, m in enumerate(mappings):
                r = []
                for j, x in enumerate(m):
                    p, inst = x
                    r.append((p, inst, rankLookup.at[inst, p.name]))
                mappings[i] = tuple(r)
            for i, m in enumerate(mappings):
                s = sum([x[2] for x in m])
                mappings[i] = (m, s)
            mappings.sort(key=lambda x: x[1])
            bestMapping = mappings[0]
            for y in bestMapping[0]:
                p, inst, _ = y
                # find corresponding duration on that best instance
                taskDurOnBestInst = realtimeLookup.at[inst, p.name]
                # make instance unavailable
                availableInstances.remove(inst)
                # start process with correct duration
                p.startProcess(time, dur=taskDurOnBestInst, runningOnInst=inst)
                runningTasks.append(p)
        else:
            earliest = min([p.end for p in runningTasks])
            finishedProcs = [a for a in runningTasks if a.end == earliest]
            time = earliest
            for p in finishedProcs:
                runningTasks.remove(p)
                availableInstances.append(p.runningOnInst)  # make instance the task ran on available again
                workflowGraph.remove_node(p.name)
        trace.append(runningTasks.copy())
    trace = [
        t for t in trace if len(t) > 0
    ]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    return time, trace


class RandomSchedulerV1(Scheduler):
    def dispatch(self, readyProcs: List[Process]):
        for p in readyProcs:
            avInst = [i for i in self.availableInstances if i.hasRoomForMoreProc]
            # find best instance
            if len(avInst) > 0:
                self.runningTasks.append(p)
                bestInst = self.pick(avInst)
                # find corresponding duration on that best instance
                taskDurOnBestInst = self.realtimeLookup.at[bestInst.id, p.name]
                # make instance unavailable
                # self.availableInstances.remove(bestInst)
                # start process with correct duration
                p.startProcess(self.time, dur=taskDurOnBestInst, runningOnInst=bestInst)
            else:
                break

    def pick(self, avInst):
        return random.choice(avInst)


def randomSchedulerV1(workflowGraph: NX.DiGraph, instances: List[int], realtimeLookup):
    time: float = 0.0
    runningTasks: List[Process] = list()
    availableInstances: List[int] = instances.copy()
    trace: List[List[Process]] = list()
    #
    nodeToProcess = {n: Process(n) for n in workflowGraph.nodes()}
    processes = list(nodeToProcess.values())
    # schedule = []
    while len(list(workflowGraph.nodes())) > 0:
        ready: List[int] = [
            t[0] for t in list(workflowGraph.in_degree()) if t[1] == 0
        ]  # all tasks that are ready to run
        ready: List[int] = [
            r for r in ready if r not in [p.name for p in runningTasks]
        ]  # all ready tasks that aren't running yet
        if len(ready) > 0 and len(availableInstances) > 0:
            readyProcs: List[Process] = [nodeToProcess[r] for r in ready]
            for p in readyProcs:
                # find best instance
                if len(availableInstances) > 0:
                    runningTasks.append(p)
                    bestInst = random.choice(availableInstances)
                    # find corresponding duration on that best instance
                    taskDurOnBestInst = realtimeLookup.at[bestInst, p.name]
                    # make instance unavailable
                    availableInstances.remove(bestInst)
                    # start process with correct duration
                    p.startProcess(time, dur=taskDurOnBestInst, runningOnInst=bestInst)  # TODO
                else:
                    break
            trace.append(runningTasks.copy())
        else:
            earliest = min([p.end for p in runningTasks])
            finishedProcs = [a for a in runningTasks if a.end == earliest]
            time = earliest
            for p in finishedProcs:
                runningTasks.remove(p)
                availableInstances.append(p.runningOnInst)  # make instance the task ran on available again
                workflowGraph.remove_node(p.name)
            trace.append(runningTasks.copy())
    # CL.print(time)
    trace = [
        t for t in trace if len(t) > 0
    ]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    # for i, t in enumerate(trace):
    #     CL.print(f"{i}\t", end='')
    #     CL.print(t)
    return time, trace


def multipleRandomSchedulerV1Execs(workflowGraph: NX.DiGraph, instances: List[int], realtimeLookup, n=1000):
    res = []
    for _ in range(n):
        res.append(randomSchedulerV1(workflowGraph.copy(), instances.copy(), realtimeLookup))
    res.sort(key=lambda x: x[0])
    times = [x[0] for x in res]
    if len(times) > 10:
        indices = [int(round(len(times) - 1 - x * (len(times) - 1), 0)) for x in [a / 10 for a in range(0, 11)]]
        return [x for i, x in enumerate(times) if i in indices]
    else:
        return times


class RoundRobinScheduler(RandomSchedulerV1):
    def __init__(self, workflowGraph: NX.DiGraph, instances: List[Instance], rankLookup, realtimeLookup):
        super().__init__(workflowGraph, instances, rankLookup, realtimeLookup)
        random.shuffle(self.availableInstances)
        self.robin = 0

    def incRobin(self):
        self.robin = (self.robin + 1) % len(self.availableInstances)

    def pick(self, avInst):
        choice = self.availableInstances[self.robin]
        self.incRobin()
        while not choice.hasRoomForMoreProc:
            choice = self.availableInstances[self.robin]
            self.incRobin()
        return choice


class FairScheduler(RandomSchedulerV1):
    def pick(self, avInst):
        lowestLoad = min(i.numRunning for i in avInst)
        return random.choice([i for i in avInst if i.numRunning == lowestLoad])


def getMethods():
    return [
        recommenderSchedulerV1,
        recommenderSchedulerV1H1,
        recommenderSchedulerV1H2,
        recommenderSchedulerV1H3,
        recommenderSchedulerV2,
        recommenderSchedulerV2H1,
        recommenderSchedulerV2H2,
        recommenderSchedulerV2H3,
    ]


def scheduleCluster(cluster, rankLookups, realtimeLookups, wfGraphs, wfNames, numRandomExecs=1000):
    instances = cluster
    #
    methods = getMethods()
    #
    times = {}
    traces = {}
    for wfName in wfNames:
        rankLookup = rankLookups[wfName]
        #
        realtimeLookup = realtimeLookups[wfName]
        #
        workflowGraph: NX.DiGraph = wfGraphs[wfName]
        #
        wfTimes = {}
        wfTraces = {}
        for fun in methods:
            time, trace = fun(workflowGraph.copy(), instances.copy(), rankLookup, realtimeLookup)
            wfTimes[fun.__name__] = time
            wfTraces[fun.__name__] = trace
        wfTimes[randomSchedulerV1.__name__] = multipleRandomSchedulerV1Execs(
            workflowGraph.copy(), instances.copy(), realtimeLookup, numRandomExecs
        )
        # CL.print(wfTimes)
        # CL.print(max([max([len(x) for x in t]) for t in wfTraces.values()]))  # find highest amount of concurrently used instances across all WFs and scheduling methods
        times[wfName] = wfTimes
        traces[wfName] = wfTraces
    return times, traces


def sanitycheckAllClusters(
    clusterData: dict = None, methods: list = None, wfs: list = None, compare: dict = None, checkAll=True, rs=None,
):
    badnesses = {
        "sanityFail": 10_000,
        "missingCluster": 1_000,
        "missingWorkflow": 1_000,
        "missingMethod": 1_000,
        "differentRandomResults": 0,
        "differentResults": 100,
    }
    badnessesLog = {
        "sanityFail": "Failed [bold red]sanity check[/] on a cluster",
        "missingCluster": "A [bold red]cluster[/] was missing",
        "missingWorkflow": "A [bold red]workflow[/] was missing",
        "missingMethod": "A [bold red]method[/] was missing",
        "differentRandomResults": "[bold red]Random results[/] differed by more than tolerable",
        "differentResults": "[bold red]Results[/] differed by more than tolerable",
    }
    badness = 0
    report = {}

    def addBadness(key, default, log=not checkAll, fields=None):
        nonlocal badness
        if log:
            rc.log(badnessesLog[key])
        if checkAll:
            badness += badnesses[key]
            prev = report.get(key, None)
            if prev is None:
                report[key] = []
            report[key].append(fields)
            return None
        else:
            return default

    for c in clusterData.values():
        if not sanitycheckCluster(c, methods, wfs):
            if (a := addBadness("sanityFail", False)) is not None:
                return a
    #
    tol = 1
    if compare is not None:
        for c, v in compare.items():
            other: dict = clusterData.get(c, None)
            if other is None:
                if (a := addBadness("missingCluster", False)) is not None:
                    return a
            else:
                for wfName, wf in v.items():
                    otherwf: dict = other.get(wfName, None)
                    if otherwf is None:
                        if (a := addBadness("missingWorkflow", False)) is not None:
                            return a
                    else:
                        for methodName, res in wf.items():
                            otherRes = otherwf.get(methodName, None)
                            if otherRes is None:
                                if (a := addBadness("missingMethod", False)) is not None:
                                    return a
                            else:
                                if type(res) is list:
                                    same = commons.list_compare(res, otherRes)
                                    if not same:
                                        diffs = [abs(r - oR) / r * 100 for r, oR in zip(res, otherRes)]
                                        acceptable = all([d <= tol for d in diffs])
                                        if not acceptable:
                                            for r, oR in zip(res, otherRes):
                                                ok = abs(r - oR) / r * 100 <= tol
                                                if not ok:
                                                    if (
                                                        a := addBadness("differentRandomResults", False, False,)
                                                    ) is not None:
                                                        return a
                                            if (
                                                a := addBadness(
                                                    "differentRandomResults",
                                                    False,
                                                    fields=[
                                                        str(statistics.mean(diffs)),
                                                        str(sum([0 if d <= tol else 1 for d in diffs])),
                                                    ],
                                                )
                                            ) is not None:
                                                return a
                                else:
                                    same = res == otherRes
                                    if not same:
                                        diff = abs(res - otherRes) / res * 100
                                        acceptable = diff <= tol
                                        if not acceptable:
                                            if (
                                                a := addBadness(
                                                    "differentResults",
                                                    False,
                                                    fields=[methodName, len(c), str(res), str(otherRes), str(diff),],
                                                )
                                            ) is not None:
                                                return a
    if badness == 0:
        rc.log("Passed full sanity check", style="bold green")
    else:
        if checkAll:
            for k, r in report.items():
                if len(r) > 0:
                    rc.log(f"{len(r)}x {badnessesLog[k]}")
                    t = rich.table.Table()
                    r = [i for i in r if i is not None]
                    trunced = False
                    if len(r) > 10:
                        r = random.sample(r, 10)
                        trunced = True
                    for f in r:
                        if f is not None:
                            t.add_row(*f)
                    if trunced and len(r) > 0:
                        t.add_row(*(["..."] * len(r[0])))
                    rc.log(t)
        rc.log(f"Failed full sanity check with {badness=}", style="bold white on red")
    return badness == 0


def sanitycheckCluster(clusterData: dict = None, methods: list = None, wfs: list = None):
    # rc.log("checking cluster")
    # rc.log(rich.panel.Panel(rich.pretty.Pretty(locals(), max_length=2)))
    #
    assert methods is not None, "methods must not be None"
    assert type(methods) is list, "methods must be a list"
    assert wfs is not None, "wfs must not be None"
    assert type(wfs) is list, "wfs must be a list"
    #
    if clusterData is None:
        rc.log("cluster data was None")
        return False
    assert type(clusterData) is dict, "clusterData must be a dict"
    for wf in wfs:
        if wf not in clusterData.keys():
            rc.log(f"cluster data was missing workflow {wf}")
            return False
    if len(wfs) != len(clusterData.items()):
        rc.log(f"cluster data had incorrect number of workflows: {len(wfs)=} != {len(clusterData.items())=}")
        return False
    #
    for wfName, wf in clusterData.items():
        for m in methods:
            if m.__name__ not in wf.keys():
                rc.log(f"cluster data was missing method {m.__name__}")
                return False
        if len(wf.items()) != len(methods) + 1:
            rc.log(f"cluster data had incorrect number of methods: {len(wf.items())=} != {len(methods)+1=}")
            return False
        #
        for methodName, res in wf.items():
            if res is None:
                rc.log("cluster data contained a None result")
                return False
            if type(res) is list:
                correctRandomRes = len(res) == 11
                if not correctRandomRes:
                    rc.log(f"cluster data had too few random results {len(res)=} != 11")
                return correctRandomRes
            else:
                return True
    rc.log(f"cluster data triggered default case")
    return False


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("targetNumClusters", type=int, action="store", default=100, nargs="?")
    argp.add_argument("--numRandomExecs", type=int, action="store", default=1000)
    argp.add_argument("--csv", type=str, action="store", dest="csvPath")
    argp.add_argument("--regModel", action="store", dest="regModelPath", default="best")
    # argp.add_argument('--extend', action='store_true', dest='extend', default=False)
    argp.add_argument("--saveLoc", type=str, action="store", dest="saveLoc", default="\0")
    argp.add_argument(
        "--saveSuffix", type=str, action="store", dest="saveSuffix", default="recSchedTimes",
    )
    argp.add_argument("--noRandom", action="store_true", dest="noRandom", default=False)
    argp.add_argument("--fullSanityCheck", action="store_true", dest="fullSanityCheck", default=False)
    cliArgs = argp.parse_args(sys.argv[1:])
    if cliArgs.regModelPath == "best":
        cliArgs.regModelPath = "models/bestModel.pickle"
    #
    if cliArgs.saveLoc == "\0":
        cliArgs.saveLoc = re.match("(.*/)*(.*?).pickle", cliArgs.regModelPath).group(2)
    pFile = Path(cliArgs.saveLoc)
    if not pFile.exists():
        pFile.mkdir()
    if pFile.is_dir():
        pFile = pFile.joinpath(re.match("(.*/)*(.*?).pickle", cliArgs.regModelPath).group(2))
        cliArgs.saveLoc = pFile.as_posix()
    #
    if cliArgs.noRandom:
        random.seed(0)
    #
    try:
        with open(cliArgs.regModelPath, "br") as f:
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
        rc.log(f"Could not load regModel from pickle at {cliArgs.regModelPath!r} with error: {e}")
        rc.print_exception()
        exit(1)
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
    wfNames = list(predBase["wfName"].unique())
    scale = preprocessing.StandardScaler().fit(
        predBase.drop(["wfName", "taskName", "nodeConfig", "realtime", "rank"], axis=1)
    )
    poly = preprocessing.PolynomialFeatures(degree=polyDeg, interaction_only=True, include_bias=polyDeg > 1).fit(
        predBase.drop(["wfName", "taskName", "nodeConfig", "realtime", "rank"], axis=1)
    )  # TODO: depends on model used!
    scale2 = preprocessing.StandardScaler().fit(
        poly.transform(
            scale.transform(predBase.drop(["wfName", "taskName", "nodeConfig", "realtime", "rank"], axis=1))
        )
    )  # TODO: pickle these once on AWS
    # valid instances: [165, 193] \ {174,177}  # c5.- (174) and c5a.large (177) are out
    allInstances = list(range(165, 194))
    allInstances.remove(174)
    allInstances.remove(177)
    #
    rankLookups = dict()
    realtimeLookups = dict()
    wfGraphs = dict()
    for wfName in wfNames:
        predBaseFilt = predBase.query("(wfName == @wfName)")
        predBaseFilt = predBaseFilt.reset_index(drop=True)
        predRankRes = regModel.predict(
            scale2.transform(
                poly.transform(
                    scale.transform(
                        predBaseFilt.drop(["wfName", "taskName", "nodeConfig", "realtime", "rank"], axis=1,)
                    )
                )
            )
        )
        predRankRes = pds.DataFrame(predRankRes, columns=["rank"])
        predRankRes = pds.concat([predBaseFilt[["taskName", "nodeConfig"]], predRankRes], axis=1)
        rankLookup = pds.DataFrame(index=list(set(allInstances)), columns=predBaseFilt["taskName"].unique())
        for row in predRankRes.itertuples():
            rankLookup.at[row.nodeConfig, row.taskName] = row.rank
        rankLookups[wfName] = rankLookup
        #
        predRealtime = predBaseFilt[["nodeConfig", "taskName", "realtime"]]
        realtimeLookup = pds.DataFrame(index=list(set(allInstances)), columns=predBaseFilt["taskName"].unique())
        for row in predRealtime.itertuples():
            realtimeLookup.at[row.nodeConfig, row.taskName] = row.realtime
        realtimeLookups[wfName] = realtimeLookup
        #
        wfShortName = re.compile("nfcore/(\w+):.*").match(wfName).group(1)
        # CL.print(f"{'-=' * 5}{wfShortName:^{len(wfShortName) + 4}}{'=-' * 5}")
        #
        workflowGraph: NX.DiGraph = NX.drawing.nx_agraph.read_dot(f"dot/{wfShortName}_1k.dot")
        wfGraphs[wfName] = workflowGraph
    # clean up
    del (
        predBaseFilt,
        predRankRes,
        rankLookup,
        row,
        predRealtime,
        realtimeLookup,
        wfShortName,
        workflowGraph,
    )
    del predBase
    del (
        name,
        regModel,
        test_confidence,
        polyDeg,
        full_confidence,
        unknown_confidence,
        train_confidence,
        bonusPickleInfo,
        loaded,
    )
    #
    with commons.stdProgress(rc) as prog:
        clusters = dict()
        genClusterProg = prog.add_task(
            f"Generating {cliArgs.targetNumClusters} different clusters", total=cliArgs.targetNumClusters,
        )
        while (
            len(clusters.keys()) < cliArgs.targetNumClusters
        ):  # TODO this does not take into consideration that --targetNumClusters may exceed the maximum number of generateable clusters
            inst = [random.choice(allInstances) for _ in range(random.randint(2, 20))]
            inst.sort()
            clusters[tuple(inst)] = inst
            prog.update(genClusterProg, completed=len(clusters.keys()))
    clusters = list(clusters.values())
    #
    times = {}
    traces = {}

    #
    def dumpResults():  # TODO this needs to perform a sanit-check on the data that is to be dumped since this also runs on error so the results may be corrupted!
        rc.log(f"Going to save to {pFile}")
        with open(pFile, "bw") as f2:
            pickle.dump(times, f2)
        rc.log(f"Saved to {pFile}")

    atexit.register(dumpResults)
    #
    pFile = Path(f"{cliArgs.saveLoc}.{cliArgs.saveSuffix}.pickle")
    #
    with commons.stdProgress(rc) as prog:
        simProg = prog.add_task(f"Scheduling on {len(clusters)} different clusters", total=len(clusters))

        #
        def makeCallback(cluster_):
            def cllbck(res):
                clusterTimes, clusterTraces = res
                times[tuple(cluster_)] = clusterTimes
                # traces[tuple(cluster_)] = clusterTraces
                prog.advance(simProg)

            return cllbck

        if pFile.is_file():
            rc.log(f"Extending previous results at {pFile}")
            with open(pFile, "br") as f:
                prevRes: dict = pickle.load(f)
            # rc.log(rich.panel.Panel(rich.pretty.Pretty(prevRes, max_length=2)))
            prevRes = {tuple(sorted(list(k))): v for k, v in prevRes.items()}
            # rc.log(rich.panel.Panel(rich.pretty.Pretty(prevRes, max_length=2)))
            # dict.fromkeys([tuple(sorted(list(k))) for k in prevRes.keys()], list(prevRes.values()))  # make sure the keys are sorted clusters
            prevKnownClusters = 0
            with Pool() as pool:
                for cluster in clusters:
                    prev = prevRes.get(tuple(cluster), None)
                    if prev is None or not sanitycheckCluster(prev, getMethods(), wfNames):
                        # CL.print(f"+ unknown cluster {cluster}")
                        pool.apply_async(
                            scheduleCluster,
                            (cluster, rankLookups, realtimeLookups, wfGraphs, wfNames, cliArgs.numRandomExecs,),
                            callback=makeCallback(cluster),
                        )
                    else:
                        times[tuple(cluster)] = prev
                        prevKnownClusters += 1
                        prog.advance(simProg)
                del prevRes, prev  # clean up
                pool.close()
                pool.join()
                rc.log(
                    f"{prevKnownClusters} known clusters and {cliArgs.targetNumClusters - prevKnownClusters} new results"
                )
        else:
            rc.log(f"No previous results found at {pFile}")
            with Pool() as pool:
                for cluster in clusters:
                    pool.apply_async(
                        scheduleCluster,
                        (cluster, rankLookups, realtimeLookups, wfGraphs, wfNames, cliArgs.numRandomExecs,),
                        callback=makeCallback(cluster),
                    )
                pool.close()
                pool.join()
    #
    if cliArgs.fullSanityCheck:
        with commons.stdProgress(rc) as prog:
            sanityComp = {}

            def makeSanityCallback(cluster_):
                def cllbck(res):
                    clusterTimes, clusterTraces = res
                    sanityComp[tuple(cluster_)] = clusterTimes
                    # traces[tuple(cluster_)] = clusterTraces
                    prog.advance(sanityProg)

                return cllbck

            #
            sanityProg = prog.add_task("Full Sanity Check", total=max(commons.iround(len(clusters) * 1 / 100), 1),)
            with Pool() as pool:
                for cluster in random.sample(clusters, max(commons.iround(len(clusters) * 1 / 100), 1)):
                    pool.apply_async(
                        scheduleCluster,
                        (cluster, rankLookups, realtimeLookups, wfGraphs, wfNames, cliArgs.numRandomExecs,),
                        callback=makeSanityCallback(cluster),
                    )
                pool.close()
                pool.join()
            sanitycheckAllClusters(times, getMethods(), wfNames, sanityComp)
    #
    dumpResults()
    atexit.unregister(dumpResults)


if __name__ == "__main__":
    # with cProfile.Profile() as pr:
    main()
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.dump_stats('perf.prof')
