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


def main():
    res = db_actions.select("*", "help2")
    transposed = {'nodeConfig'         : [165, 166, 167, 168, 169, 170, 171, 172, 173, 175, 176, 178, 179, 180, 181,
                                          182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193],
                  'build-linux-kernel1': [],
                  'fio2'               : [],
                  'fio3'               : [],
                  'fio4'               : [],
                  'fio5'               : [],
                  'fio6'               : [],
                  'fio7'               : [],
                  'fio8'               : [],
                  'fio9'               : [],
                  'iperf10'            : [],
                  'iperf11'            : [],
                  'iperf12'            : [],
                  'iperf13'            : [],
                  'john-the-ripper14'  : [],
                  'john-the-ripper15'  : [],
                  'ramspeed16'         : [],
                  'ramspeed17'         : [],
                  'ramspeed18'         : [],
                  'ramspeed19'         : [],
                  'ramspeed20'         : [],
                  'ramspeed21'         : [],
                  'ramspeed22'         : [],
                  'ramspeed23'         : [],
                  'ramspeed24'         : [],
                  'ramspeed25'         : [],
                  'stream26'           : [],
                  'stream27'           : [],
                  'stream28'           : [],
                  'stream29'           : []}
    cols = ['build-linux-kernel1', 'fio2', 'fio3', 'fio4', 'fio5', 'fio6', 'fio7', 'fio8', 'fio9', 'iperf10', 'iperf11',
            'iperf12', 'iperf13', 'john-the-ripper14', 'john-the-ripper15', 'ramspeed16', 'ramspeed17', 'ramspeed18',
            'ramspeed19', 'ramspeed20', 'ramspeed21', 'ramspeed22', 'ramspeed23', 'ramspeed24', 'ramspeed25',
            'stream26', 'stream27', 'stream28', 'stream29']
    for r in res:
        print(r[4], r[5], r[6])
    for i in range(0, len(res[0][6])):
        for r in res:
            for col in cols:
                if r[4] == col:
                    transposed[col].append(r[6][i])
    # pprint(transposed)

    # print('create table "nodeBenchmarkTransposedRankings" ( "nodeConfig" int,' + ','.join(        ['"' + c + '"' + ' int' for c in cols]) + ');')
    rows = []
    for i, node in enumerate(transposed['nodeConfig']):
        row = [node]
        for col in cols:
            row.append(transposed[col][i])
        rows.append(data_types.node_benchmark_transposed_rankings_entry(*row))

    if False:
        for row in rows:
            db_actions.insert(row, "nodeBenchmarkTransposedRankings", underscoreToDash=True, retID=False)


if __name__ == '__main__':
    main()
