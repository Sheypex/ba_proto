import cProfile
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
        self.runningOnInst = None

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
        # print(f"Started {self!r}")

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
        ready: List[int] = [t[0] for t in list(workflowGraph.in_degree()) if
                            t[1] == 0]  # all tasks that are ready to run
        ready: List[int] = [r for r in ready if
                            r not in [p.name for p in runningTasks]]  # all ready tasks that aren't running yet
        if len(ready) > 0 and len(availableInstances) > 0:
            readyProcs: List[Process] = [nodeToProcess[r] for r in ready]
            for p in readyProcs:
                # find best instance
                if len(availableInstances) > 0:
                    runningTasks.append(p)
                    predictedRanks = {}
                    for inst in list(set(availableInstances)):
                        predictedRanks[inst] = rankLookup.at[inst, p.name]
                    bestInst: int = [n for n, r in predictedRanks.items() if
                                     r == min([pr for pr in predictedRanks.values()])][
                        0]  # TODO: this arbitrarily chooses the first instance among all that have the highest rank # TODO: should the predicted ranks be rounded??
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
    trace = [t for t in trace if
             len(t) > 0]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    return time, trace


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
        ready: List[int] = [t[0] for t in list(workflowGraph.in_degree()) if
                            t[1] == 0]  # all tasks that are ready to run
        ready: List[int] = [r for r in ready if
                            r not in [p.name for p in runningTasks]]  # all ready tasks that aren't running yet
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
                    bestInst: int = [n for n, r in predictedRanks.items() if
                                     r == min([pr for pr in predictedRanks.values()])][
                        0]  # TODO: this arbitrarily chooses the first instance among all that have the highest rank # TODO: should the predicted ranks be rounded??
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
    trace = [t for t in trace if
             len(t) > 0]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    return time, trace


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
        ready: List[int] = [t[0] for t in list(workflowGraph.in_degree()) if
                            t[1] == 0]  # all tasks that are ready to run
        ready: List[int] = [r for r in ready if
                            r not in [p.name for p in runningTasks]]  # all ready tasks that aren't running yet
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
                    bestInst: int = [n for n, r in predictedRanks.items() if
                                     r == min([pr for pr in predictedRanks.values()])][
                        0]  # TODO: this arbitrarily chooses the first instance among all that have the highest rank # TODO: should the predicted ranks be rounded??
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
    trace = [t for t in trace if
             len(t) > 0]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    return time, trace


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
    sinks = [t[0] for t in list(workflowGraph.out_degree()) if
             t[1] == 0]

    def tmpSort(x):
        if x not in sinks:
            return -max([len(a) for a in list(NX.algorithms.all_simple_paths(workflowGraph, x, sinks))])
        else:
            return 0

    #
    while len(list(workflowGraph.nodes())) > 0:
        ready: List[int] = [t[0] for t in list(workflowGraph.in_degree()) if
                            t[1] == 0]  # all tasks that are ready to run
        ready: List[int] = [r for r in ready if
                            r not in [p.name for p in runningTasks]]  # all ready tasks that aren't running yet
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
                    bestInst: int = [n for n, r in predictedRanks.items() if
                                     r == min([pr for pr in predictedRanks.values()])][
                        0]  # TODO: this arbitrarily chooses the first instance among all that have the highest rank # TODO: should the predicted ranks be rounded??
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
    # print(time)
    trace = [t for t in trace if
             len(t) > 0]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    # for i, t in enumerate(trace):
    #     print(f"{i}\t", end='')
    #     pprint(t)
    return time, trace


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
        ready: List[int] = [t[0] for t in list(workflowGraph.in_degree()) if
                            t[1] == 0]  # all tasks that are ready to run
        ready: List[int] = [r for r in ready if
                            r not in [p.name for p in runningTasks]]  # all ready tasks that aren't running yet
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
    trace = [t for t in trace if
             len(t) > 0]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    return time, trace


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
        ready: List[int] = [t[0] for t in list(workflowGraph.in_degree()) if
                            t[1] == 0]  # all tasks that are ready to run
        ready: List[int] = [r for r in ready if
                            r not in [p.name for p in runningTasks]]  # all ready tasks that aren't running yet
        if len(ready) > 0 and len(availableInstances) > 0:
            ready.sort(key=lambda x: -workflowGraph.out_degree(x))  # H1
            readyProcs: List[Process] = [nodeToProcess[r] for r in ready]
            procComb = [readyProcs[0:min([len(availableInstances), len(readyProcs)])]]
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
    trace = [t for t in trace if
             len(t) > 0]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    return time, trace


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
        ready: List[int] = [t[0] for t in list(workflowGraph.in_degree()) if
                            t[1] == 0]  # all tasks that are ready to run
        ready: List[int] = [r for r in ready if
                            r not in [p.name for p in runningTasks]]  # all ready tasks that aren't running yet
        if len(ready) > 0 and len(availableInstances) > 0:
            ready.sort(key=lambda x: -(len(NX.algorithms.dag.descendants(workflowGraph, x))))  # H2
            readyProcs: List[Process] = [nodeToProcess[r] for r in ready]
            procComb = [readyProcs[0:min([len(availableInstances), len(readyProcs)])]]
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
    trace = [t for t in trace if
             len(t) > 0]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    return time, trace


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
    sinks = [t[0] for t in list(workflowGraph.out_degree()) if
             t[1] == 0]

    def tmpSort(x):
        if x not in sinks:
            return -max([len(a) for a in list(NX.algorithms.all_simple_paths(workflowGraph, x, sinks))])
        else:
            return 0

    #
    while len(list(workflowGraph.nodes())) > 0:
        ready: List[int] = [t[0] for t in list(workflowGraph.in_degree()) if
                            t[1] == 0]  # all tasks that are ready to run
        ready: List[int] = [r for r in ready if
                            r not in [p.name for p in runningTasks]]  # all ready tasks that aren't running yet
        if len(ready) > 0 and len(availableInstances) > 0:
            ready.sort(key=tmpSort)  # H3
            readyProcs: List[Process] = [nodeToProcess[r] for r in ready]
            procComb = [readyProcs[0:min([len(availableInstances), len(readyProcs)])]]
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
    trace = [t for t in trace if
             len(t) > 0]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    return time, trace


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
        ready: List[int] = [t[0] for t in list(workflowGraph.in_degree()) if
                            t[1] == 0]  # all tasks that are ready to run
        ready: List[int] = [r for r in ready if
                            r not in [p.name for p in runningTasks]]  # all ready tasks that aren't running yet
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
    # print(time)
    trace = [t for t in trace if
             len(t) > 0]  # strip empty runningTasks | happens whenever in one step all running tasks end and only in the next step some new task(s) are scheduled to run
    # for i, t in enumerate(trace):
    #     print(f"{i}\t", end='')
    #     pprint(t)
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


def scheduleCluster(cluster, rankLookups, realtimeLookups, wfGraphs, wfNames, numRandomExecs=1000):
    instances = cluster
    #
    methods = [recommenderSchedulerV1, recommenderSchedulerV1H1, recommenderSchedulerV1H2, recommenderSchedulerV1H3,
               recommenderSchedulerV2, recommenderSchedulerV2H1, recommenderSchedulerV2H2, recommenderSchedulerV2H3]
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
        wfTimes[randomSchedulerV1.__name__] = multipleRandomSchedulerV1Execs(workflowGraph.copy(), instances.copy(), realtimeLookup, numRandomExecs)
        # pprint(wfTimes)
        # print(max([max([len(x) for x in t]) for t in wfTraces.values()]))  # find highest amount of concurrently used instances across all WFs and scheduling methods
        times[wfName] = wfTimes
        traces[wfName] = wfTraces
    return times, traces


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('targetNumClusters', type=int, action='store', default=100, nargs='?')
    argp.add_argument('--numRandomExecs', type=int, action='store', default=1000)
    argp.add_argument('--csv', type=str, action='store', dest='csvPath')
    argp.add_argument('--regModel', action='store', dest='regModelPath', default='best')
    # argp.add_argument('--extend', action='store_true', dest='extend', default=False)
    argp.add_argument('--saveLoc', type=str, action="store", dest="saveLoc", default="\0")
    argp.add_argument('--saveSuffix', type=str, action="store", dest="saveSuffix", default="recSchedTimes")
    argp.add_argument('--noRandom', action='store_true', dest='noRandom', default=False)
    cliArgs = argp.parse_args(sys.argv[1:])
    if cliArgs.regModelPath == "best":
        cliArgs.regModelPath = 'models/bestModel.pickle'
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
        loaded = pickle.load(open(cliArgs.regModelPath, 'br'))
        _, regModel, _, polyDeg = loaded
    except Exception as e:
        print(f"Could not load regModel from pickle at {cliArgs.regModelPath!r} with error: {e}", file=sys.stderr)
        exit(1)
    if cliArgs.csvPath:
        predBase = None
        try:
            predBase = pds.read_csv(open(cliArgs.csvPath, 'r'))  # already filtered for realtime > 1000)
        except:
            pass
        try:
            predBase = pds.read_csv(open(cliArgs.csvPath + '.csv', 'r'))
        except:
            pass
        if predBase is None:
            raise Exception(f"Couldn't open csv file {cliArgs.csvPath!r} or {(cliArgs.csvPath + '.csv')!r}")
    else:
        with db_actions.connect() as conn:
            predBase = pds.read_sql("SELECT * FROM \"averageRuntimesPredictionBase1000\"", conn)
    #
    wfNames = list(predBase['wfName'].unique())
    scale = preprocessing.StandardScaler().fit(
        predBase.drop(['wfName', 'taskName', 'nodeConfig', 'realtime', 'rank'], axis=1))
    poly = preprocessing.PolynomialFeatures(degree=polyDeg, interaction_only=True, include_bias=polyDeg > 1).fit(
        predBase.drop(['wfName', 'taskName', 'nodeConfig', 'realtime', 'rank'], axis=1))  # TODO: depends on model used!
    scale2 = preprocessing.StandardScaler().fit(
        poly.transform(scale.transform(predBase.drop(['wfName', 'taskName', 'nodeConfig', 'realtime', 'rank'],
                                                     axis=1))))  # TODO: pickle these once on AWS
    # valid instances: [165, 193] \ {174,177}  # c5.- (174) and c5a.large (177) are out
    allInstances = list(range(165, 194))
    allInstances.remove(174)
    allInstances.remove(177)
    #
    rankLookups = dict()
    realtimeLookups = dict()
    wfGraphs = dict()
    for wfName in wfNames:
        predBaseFilt = predBase.query(
            "(wfName == @wfName)")
        predBaseFilt = predBaseFilt.reset_index(drop=True)
        predRankRes = regModel.predict(scale2.transform(poly.transform(scale.transform(
            predBaseFilt.drop(['wfName', 'taskName', 'nodeConfig', 'realtime', 'rank'], axis=1)))))
        predRankRes = pds.DataFrame(predRankRes, columns=['rank'])
        predRankRes = pds.concat([predBaseFilt[['taskName', 'nodeConfig']], predRankRes], axis=1)
        rankLookup = pds.DataFrame(index=list(set(allInstances)), columns=predBaseFilt['taskName'].unique())
        for row in predRankRes.itertuples():
            rankLookup.at[row.nodeConfig, row.taskName] = row.rank
        rankLookups[wfName] = rankLookup
        #
        predRealtime = predBaseFilt[['nodeConfig', 'taskName', 'realtime']]
        realtimeLookup = pds.DataFrame(index=list(set(allInstances)), columns=predBaseFilt['taskName'].unique())
        for row in predRealtime.itertuples():
            realtimeLookup.at[row.nodeConfig, row.taskName] = row.realtime
        realtimeLookups[wfName] = realtimeLookup
        #
        wfShortName = re.compile("nfcore/(\w+):.*").match(wfName).group(1)
        # print(f"{'-=' * 5}{wfShortName:^{len(wfShortName) + 4}}{'=-' * 5}")
        #
        workflowGraph: NX.DiGraph = NX.drawing.nx_agraph.read_dot(f"dot/{wfShortName}_1k.dot")
        wfGraphs[wfName] = workflowGraph
    #
    clusters = []
    with alive_bar(cliArgs.targetNumClusters, f"Generating {cliArgs.targetNumClusters} different clusters") as bar:
        while len(
                clusters) < cliArgs.targetNumClusters:  # TODO this does not take into consideration that --targetNumClusters may exceed the maximum number of generateable clusters
            inst = [random.choice(allInstances) for _ in range(random.randint(2, 20))]
            if inst not in clusters:
                clusters.append(inst)
                bar()
    times = {}
    traces = {}
    with alive_bar(len(clusters), f"Scheduling on {len(clusters)} different clusters") as bar:
        def makeCallback(cluster_):
            def cllbck(res):
                clusterTimes, clusterTraces = res
                times[tuple(cluster_)] = clusterTimes
                traces[tuple(cluster_)] = clusterTraces
                bar()

            return cllbck

        with Pool() as pool:
            for cluster in clusters:
                pool.apply_async(scheduleCluster, (cluster, rankLookups, realtimeLookups, wfGraphs, wfNames, cliArgs.numRandomExecs), callback=makeCallback(cluster))
            pool.close()
            pool.join()
    pickle.dump(times, open(f"{cliArgs.saveLoc}.{cliArgs.saveSuffix}.pickle", 'bw'))
    print(f"Saved to {cliArgs.saveLoc}.{cliArgs.saveSuffix}.pickle")
    # pickle.dump(traces, open("recSchedTraces.pickle", 'bw'))


if __name__ == "__main__":
    # with cProfile.Profile() as pr:
    main()
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.dump_stats('perf.prof')
