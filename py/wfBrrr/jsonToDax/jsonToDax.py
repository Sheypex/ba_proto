import itertools
import json
import pathlib
import re
from dataclasses import dataclass, fields
from typing import List, Dict

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
    runtimeInSeconds: float
    # runtime: float
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


def main():
    # make_json()
    #
    with open("methyl.json", "rb") as f:
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
    idTaskContentLUT = {}
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
               '@runtime': t.runtimeInSeconds * t.cores,
               '@runtime_raw': t.runtimeInSeconds,
               'uses': []
               }
        idXmlLUT[t.id] = xml
        idTaskContentLUT[t.id] = t
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
    with open("files.json", "rb") as f:
        filesContent: List[dict] = json.load(f)
    filesContent: List[FileContent] = [DataClassUnpack.instantiate(FileContent, f) for f in filesContent]
    # collect all files associated with each task name
    # task names in files.json may mismatch against instance.json task names
    baseTaskNameFilesLUT: Dict[str, List[FileContent]] = {}
    for f in filesContent:
        baseTaskNameFilesLUT.setdefault(f.taskName, [])
        baseTaskNameFilesLUT[f.taskName].append(f)
    assert all(len(set(f.id for f in v)) == 1 for k, v in baseTaskNameFilesLUT.items())

    # check if we're parsing a synthetic instance
    # Todo just checking the first should be enough but is technically risky
    res = re.fullmatch(r"(.*?)_(\d{8})", tasksContent[0].name)
    if res is None:
        synth = False
    else:
        baseName, longId = res.groups()
        synth = longId == tasksContent[0].id  # should be the case anyway but anything else would be cursed
        assert synth

    # big LUT for the files that will be associated with a task id
    taskIdFiles = {}

    # only need to take care of file cloning if instance is synthetic
    if synth:
        # 'clone' files as needed and parse files into xml
        tasksRemaining = [t.id for t in tasksContent]  # all the tasks still need their files to be setup
        while len(tasksRemaining):
            # find a task, where all parents are finished
            doNext = 0
            for t in (idTaskContentLUT[x] for x in tasksRemaining):
                if all(nameIDLUT[p] not in tasksRemaining for p in t.parents):
                    doNext = t
                    break
            else:
                raise RuntimeError("dependency cycle in tasks?")
            # finish found task
            # find base name
            baseName, longId = re.fullmatch(r"(.*?)_(\d{8})", doNext.name).groups()
            # add files of base name equivalent task to this tasks file list
            shouldFiles: List[FileContent] = baseTaskNameFilesLUT[baseName]
            for f in shouldFiles:
                taskIdFiles.setdefault(doNext.id, [])
                # output files are 'tagged' with this tasks id
                if f.output:
                    taskIdFiles[doNext.id].append(
                        FileContent(doNext.name, f"{f.fileName}_{doNext.id}", f.tag, f.size, f.output, doNext.id))

                # input files will have the id of the respective parent id as their tag, or they are global inputs
                else:
                    # find the parent this file belongs to
                    parentFile = False
                    for p in doNext.parents:
                        for bf in baseTaskNameFilesLUT[re.fullmatch(r"(.*?)_(\d{8})", p).group(1)]:
                            if not bf.output:
                                continue
                            if f.fileName == bf.fileName:
                                parentFile = True
                                # use the parents output as tagged by their id as input here
                                taskIdFiles[doNext.id].append(
                                    FileContent(doNext.name, f"{f.fileName}_{nameIDLUT[p]}", f.tag, f.size, f.output,
                                                doNext.id))

                    # is it a global input file? i.e. were no parents found that have this file as an output?
                    if not parentFile:
                        # use the global file of the given file name as input instead
                        taskIdFiles[doNext.id].append(
                            FileContent(doNext.name, f.fileName, f.tag, f.size, f.output, doNext.id))

            # update remaining
            tasksRemaining.remove(doNext.id)

    # otherwise just conform to format for output
    else:
        for f in filesContent:
            taskIdFiles.setdefault(f.id, [])
            taskIdFiles[f.id].append(f)

    # output files as formatted to the xml
    for tId, files in taskIdFiles.items():
        # <uses file="SRR7961164_GSM3415728_MShef4_J3_SRR7961164_GSM3415728_MShef4_J3_1_val_1_bismark_bt2_pe.deduplicated.bam"
        #               link="input" size="3942292891"/>
        for f in files:
            idXmlLUT[tId]["uses"].append({
                "@file": f.fileName,
                "@link": "output" if f.output else "input",
                "@size": f.size
            })

    # done, output xml
    print(xmltodict.unparse(output, pretty=True))


def make_json():
    parser = wfcommons.wfinstances.NextflowLogsParser(execution_dir=pathlib.Path("pipeline_info"))
    wf = parser.build_workflow("methylseq")
    wf.write_json(pathlib.Path("methylseq.json"))


if __name__ == '__main__':
    main()
