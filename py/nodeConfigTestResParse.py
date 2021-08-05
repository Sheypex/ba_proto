import json
import os
import re
from pprint import pprint

import data_types
from my_yaml import yaml_load, yaml_dump

# CONSTANTS
resultsDir = "../cloud/aws/results/"
tests = ['fio', 'ramspeed', 'stream', 'iperf', 'build-linux-kernel', 'john-the-ripper']


def main():
    # gather json res
    results = {}
    for resFile in os.listdir(resultsDir):
        results[resFile] = json.load(open(f"{resultsDir}{resFile}/nodeconfigtestres.json", "r"))['results']
    json.dump(results, open('results.json', 'w'), indent=2)
    # filter by test
    filtered = {}
    for testName in tests:
        filtered[testName] = {}
        rsearch = re.compile(testName)
        for instanceType, res in results.items():
            found = []
            for test in res:
                searchRes = rsearch.search(test['test'])
                if searchRes:
                    found.append(test)
            filtered[testName][instanceType] = found
    json.dump(filtered, open('filtered.json', 'w'), indent=2)
    # find missing
    missing = {}
    for testName, instances in filtered.items():
        missing[testName] = []
        for instanceType, testRes in instances.items():
            if not len(testRes):
                missing[testName].append(instanceType)
        if not len(missing[testName]):
            del missing[testName]
    pprint(missing)
    json.dump(missing, open('missing.json','w'), indent=2)

if __name__ == '__main__':
    main()
