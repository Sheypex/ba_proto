from pprint import pprint

import networkx as NX

from db_actions import db_actions

import pandas as pds
import matplotlib.pyplot as plt


def main():
    wfNames = ['chipseq', 'eager', 'methylseq', 'sarek', 'viralrecon']
    with db_actions.connect() as conn:
        dF = pds.read_sql(
            "select distinct \"taskName\", \"wfName\" from \"averageRuntimesPredictionBase1000\"",
            conn)
    for wfName in wfNames:
        wfTasks = dF[dF.wfName.str.contains(wfName)]
        # print(wfTasks)
        wfDot = NX.drawing.nx_agraph.read_dot(f"dot/{wfName}.dot")
        # print(wfDot)
        nodes = list(wfDot.nodes(data='label'))
        nodes = [n for n in nodes if n[1] is not None and n[1] in list(wfTasks.taskName.values)]
        assert all([task in [n[1] for n in nodes] for task in list(wfTasks.taskName.values)])
        anc = {}
        for n in nodes:
            node, task = n
            anc[node] = [d for d in NX.algorithms.dag.ancestors(wfDot, node) if d in [n[0] for n in nodes]]
        dep = {}
        for n in anc.keys():
            task = [no[1] for no in nodes if no[0] == n][0]
            a = anc[n]
            dep[task] = [[no[1] for no in nodes if no[0] == d][0] for d in a]
        # pprint(dep)
        expo = NX.DiGraph()
        edges = []
        for a in dep.keys():
            for b in dep[a]:
                edges.append((b, a))
        expo.add_nodes_from(dep.keys())
        expo.add_edges_from(edges)
        NX.drawing.nx_agraph.write_dot(expo, f"dot/{wfName}_1k.dot")


if __name__ == "__main__":
    main()
