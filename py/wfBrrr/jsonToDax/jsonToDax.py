import itertools
import json
import pathlib
from dataclasses import dataclass, fields
from typing import List

import rich
import wfcommons
import xmltodict


class DataClassUnpack:
    classFieldCache = {}

    @classmethod
    def instantiate(cls, classToInstantiate, argDict):
        if classToInstantiate not in cls.classFieldCache:
            cls.classFieldCache[classToInstantiate] = {f.name for f in fields(classToInstantiate) if f.init}

        fieldSet = cls.classFieldCache[classToInstantiate]
        filteredArgDict = {k: v for k, v in argDict.items() if k in fieldSet}
        return classToInstantiate(**filteredArgDict)


@dataclass
class TaskContent:
    name: str
    # type: str
    # command: dict
    parents: List[str]
    children: List[str]
    # files: List[str]
    id: str
    # "category": "nfcore_methylseq:methylseq:input_check:samplesheet_check"
    # runtimeInSeconds: float
    runtime: float
    cores: float
    # avgCPU: float
    # readBytes: int
    # writtenBytes: int
    # memoryInBytes: int


@dataclass
class FileContent:
    taskName: str
    fileName: str
    tag: str
    size: int
    output: bool
    id: str


def make_json():
    parser = wfcommons.wfinstances.NextflowLogsParser(execution_dir=pathlib.Path("pipeline_info"))
    wf = parser.build_workflow("methylseq")
    wf.write_json(pathlib.Path("methylseq.json"))


def main():
    # make_json()
    #
    with open("test.json", "rb") as f:
        jsonContent = json.load(f)
    tasksContent: List[TaskContent] = [DataClassUnpack.instantiate(TaskContent, t) for t in
                                       jsonContent["workflow"]["tasks"]]
    output = {
        "adag": {
            "@xmlns": "http://pegasus.isi.edu/schema/DAX",
            "@xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "@version": "2.1",
            "@xsi:schemaLocation": "http://pegasus.isi.edu/schema/DAX http://pegasus.isi.edu/schema/dax-2.1.xsd",
            "job": [],
            "child": [],
        }
    }
    # name <-> id LUT
    nameIDLUT = {}
    for t in tasksContent:
        assert t.name not in nameIDLUT
        assert t.id not in nameIDLUT
        nameIDLUT[t.name] = t.id
        nameIDLUT[t.id] = t.name
    # id <-> xmldict LUT
    idXmlLUT = {}
    # create job definitions
    # <job id="1" name="NFCORE_METHYLSEQ:METHYLSEQ:INPUT_CHECK:SAMPLESHEET_CHECK (samplesheet_full.csv)"
    #          namespace="methylseq" numcores="1" runtime="0.0" runtime_raw="0.0">
    #         <uses file="samplesheet_full.csv_samplesheet_full.csv" link="input" size="1971"/>
    #         <uses file="task_id_1_versions.yml" link="output" size="78"/>
    #         <uses file="task_id_1_samplesheet.valid.csv" link="output" size="2053"/>
    #     </job>
    # with open("reference.dax", "rb") as f:
    #     print(xmltodict.parse(f))
    for t in tasksContent:
        xml = {'@id': t.id,
               '@name': t.name,
               # '@namespace': 'methylseq', # TODO
               '@numcores': t.cores,
               '@runtime': t.runtime * t.cores,
               '@runtime_raw': t.runtime,
               'uses': []
               }
        idXmlLUT[t.id] = xml
        output["adag"]["job"].append(xml)
    # create job dag
    for t in tasksContent:
        if len(t.parents) > 0:
            output["adag"]["child"].append({
                "@ref": t.id,
                "parent": [{
                    "@ref": nameIDLUT[p]
                } for p in t.parents]
            })
    # create file dependencies
    if False:
        with open("files.json", "rb") as f:
            filesContent = json.load(f)
        filesContent:List[FileContent] = [DataClassUnpack.instantiate(FileContent, f) for f in filesContent]
        for f in filesContent:
            # <uses file="SRR7961164_GSM3415728_MShef4_J3_SRR7961164_GSM3415728_MShef4_J3_1_val_1_bismark_bt2_pe.deduplicated.bam"
            #               link="input" size="3942292891"/>
            idXmlLUT[f.id]["uses"].append({
                "@file": f.fileName,
                "@link": "output" if f.output else "input",
                "@size": f.size
            })
    print(xmltodict.unparse(output, pretty=True))


if __name__ == '__main__':
    main()
