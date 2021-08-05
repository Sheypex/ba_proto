import dataclasses
import json
import os
import shutil
from dataclasses import dataclass
from pprint import pprint
from typing import Dict

import data_types
import testNodeConfigs
from db_actions import db_actions
from my_yaml import yaml_load, yaml_dump

# CONSTANTS
resultsDir = '../cloud/aws/results/'


@dataclass()
class TestID:
    test: str
    arguments: str
    units: str

    def __eq__(self, other):
        if isinstance(other, TestID):
            return self.test == other.test and self.arguments == other.arguments and self.units == other.units
        else:
            return False


def main():
    instanceTypes = testNodeConfigs.instanceTypes
    instanceTypeNames = [x.name for x in instanceTypes]
    instanceFamilies = testNodeConfigs.instanceFamilies
    #
    if False:
        fileNames = []
        for ITResDir in os.listdir(resultsDir):
            for file in os.listdir(resultsDir + ITResDir):
                if file not in fileNames:
                    fileNames.append(file)

        print(fileNames)

        hasAll = []
        for ITResDir in os.listdir(resultsDir):
            testNames = True
            for file in fileNames:
                if file not in os.listdir(resultsDir + ITResDir):
                    testNames = False
                    break
            if testNames:
                hasAll.append(ITResDir)
        hasAll.sort()
        pprint(hasAll)

    if False:
        for ITResDir in os.listdir(resultsDir):
            nodeconfigtestres = json.load(open(resultsDir + ITResDir + "/nodeconfigtestres.json", 'r'))
            json.dump(nodeconfigtestres, open(resultsDir + ITResDir + "/nodeconfigtestres.json", 'w'), indent=2)
            print(f"wrote {resultsDir + ITResDir + '/nodeconfigtestres.json'}")

        for ITResDir in hasAll:
            stream = json.load(open(resultsDir + ITResDir + "/stream.json", 'r'))["results"]
            nodeconfigtestres = json.load(open(resultsDir + ITResDir + "/nodeconfigtestres.json", 'r'))
            nodeconfigtestres["results"] += stream
            json.dump(nodeconfigtestres, open(resultsDir + ITResDir + "/nodeconfigtestres.json", 'w'), indent=2)
            print(f"wrote {resultsDir + ITResDir + '/nodeconfigtestres.json'}")

        for ITResDir in os.listdir(resultsDir):
            os.remove(resultsDir + ITResDir + '/complete.json')
    #
    # benchmarkResults = {
    #     name: json.load(open(resultsDir + "/" + name + "/nodeconfigtestres.json", 'r'))["results"] for name in
    #     instanceTypeNames
    # }
    benchmarkResults = json.load(open("results.json", "r"))
    # json.dump(benchmarkResults, open('results.json', 'w'), indent=2)
    #
    if False:
        testNames = []
        for name, res in benchmarkResults.items():
            for test in res:
                if test["test"] not in testNames:
                    testNames.append(test["test"])
        pprint(testNames)

        # sanity check
        for name, res in benchmarkResults.items():
            for test in testNames:
                assert test in [x["test"] for x in res]

        # rename
        testNameUpdate = {
            'pts/fio-1.14.1'               : "fio",
            'pts/ramspeed-1.4.3'           : "ramspeed",
            'pts/iperf-1.1.1'              : "iperf",
            'pts/john-the-ripper-1.7.2'    : "john-the-ripper",
            'pts/build-linux-kernel-1.11.0': "build-linux-kernel",
            'pts/stream-1.3.2'             : "stream"
        }
        for name, res in benchmarkResults.items():
            for test in res:
                newName = testNameUpdate[test["test"]]
                test["test"] = newName
        testNames = []
        for name, res in benchmarkResults.items():
            for test in res:
                if test["test"] not in testNames:
                    testNames.append(test["test"])
        print("new names:")
        pprint(testNames)
        # json.dump(benchmarkResults, open('results.json', 'w'), indent=2)  # field "arguments" changed by hand !! dont willy nilly dump again!
    #
    testIDs = []
    for name, res in benchmarkResults.items():
        for test in res:
            id = TestID(test["test"], test["arguments"], test["units"])
            if id not in testIDs:
                testIDs.append(id)
    pprint(testIDs)

    # sanity check
    for id in testIDs:
        found = False
        for name, res in benchmarkResults.items():
            for test in res:
                comp = TestID(test["test"], test["arguments"], test["units"])
                if id == comp:
                    found = True
                    break
        assert found
    #
    if False:
        json.dump(benchmarkResults, open("results.org.json", "w"), indent=2)
        for iTName, res in benchmarkResults.items():
            for test in res:
                assert len(test["results"].items()) == 1
                for _, val in test["results"].items():
                    test["result"] = val["value"]
                    del test["results"]
        json.dump(benchmarkResults, open("results.json", "w"), indent=2)
    #
    if False:
        nodeConfigIdLookup = yaml_load(open("nodeConfigIdLookup.yaml", "r"))
        testBenchmarkedComponentLookup = {
            "fio"               : data_types.node_component_enum.memory,
            "ramspeed"          : data_types.node_component_enum.ram,
            "iperf"             : data_types.node_component_enum.internet,
            "john-the-ripper"   : data_types.node_component_enum.cpu,
            "build-linux-kernel": data_types.node_component_enum.cpu,
            "stream"            : data_types.node_component_enum.ram
        }
        # compute scores
        maxBuild = max(
            [float(res["result"]) for _, x in benchmarkResults.items() for res in x if
             res["test"] == "build-linux-kernel"]
        )
        for iT, x in benchmarkResults.items():
            for res in x:
                if res["test"] == "build-linux-kernel":
                    res["score"] = round(maxBuild - float(res["result"]), 3)
                else:
                    res["score"] = float(res["result"])
        # add to DB
        for name, res in benchmarkResults.items():
            for test in res:
                assert test["test"] in testBenchmarkedComponentLookup.keys()
                db_actions.insert(
                    data_types.node_benchmarks_entry(
                        nodeConfigIdLookup[name],
                        testBenchmarkedComponentLookup[test["test"]].name,
                        test["arguments"],
                        test["result"],
                        test["test"],
                        test["units"],
                        test["score"]
                    ), data_types.db_tables_enum.nodeBenchmarks)

    pprint([len(x) for _,x in benchmarkResults.items()])


if __name__ == '__main__':
    main()
