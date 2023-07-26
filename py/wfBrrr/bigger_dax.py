import glob
import itertools
import json
import logging
import pathlib
import random
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import List, Dict, Union, Any, Iterable

import click
import rich
import rich.prompt
import xmltodict
import pandas as pds
from deepdiff import DeepDiff
from stringcase import capitalcase
from rich.logging import RichHandler
import attrs

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("rich")


def forceSequence(we: Any):
    if not isinstance(we, Sequence):
        return [we]
    return we


def nukeSolitaryList(we: Any):
    if isinstance(we, Sequence):
        if len(we) == 1:
            return we[0]
    return we


@attrs.define
class FileContent:
    file: str = attrs.field(converter=str)
    link: str = attrs.field(converter=str)
    size: int = attrs.field(converter=int)

    @classmethod
    def from_dax(cls, dax_elem):
        return cls(dax_elem["@file"], dax_elem["@link"], dax_elem["@size"])

    def to_dax(self):
        return {
            "@file": str(self.file),
            "@link": str(self.link),
            "@size": str(self.size)
        }


@attrs.define
class JobContent:
    id: str = attrs.field(converter=str)
    name: str = attrs.field(converter=str)
    namespace: str = attrs.field(converter=str)
    numcores: int = attrs.field(converter=int)
    runtime: float = attrs.field(converter=float)
    runtime_raw: float = attrs.field(converter=float)
    uses: List[FileContent] = attrs.Factory(list)

    @classmethod
    def from_dax(cls, dax_elem):
        return cls(dax_elem["@id"], dax_elem["@name"], dax_elem["@namespace"], dax_elem["@numcores"], dax_elem["@runtime"], dax_elem["@runtime_raw"],
                   [FileContent.from_dax(sub_elem) for sub_elem in forceSequence(dax_elem["uses"])])

    def to_dax(self):
        return {
            "@id"         : str(self.id),
            "@name"       : str(self.name),
            "@namespace"  : str(self.namespace),
            "@numcores"   : str(self.numcores),
            "@runtime"    : str(self.runtime),
            "@runtime_raw": str(self.runtime_raw),
            "uses"        : nukeSolitaryList([sub_elem.to_dax() for sub_elem in self.uses])
        }


@attrs.define
class ParentContent:
    ref: str = attrs.field(converter=str)

    @classmethod
    def from_dax(cls, dax_elem):
        return cls(dax_elem["@ref"])

    def to_dax(self):
        return {
            "@ref": str(self.ref)
        }


@attrs.define
class ChildContent:
    ref: str = attrs.field(converter=str)
    parent: List[ParentContent] = attrs.Factory(list)

    @classmethod
    def from_dax(cls, dax_elem):
        return cls(dax_elem["@ref"], [ParentContent.from_dax(sub_elem) for sub_elem in forceSequence(dax_elem["parent"])])

    def to_dax(self):
        return {
            "@ref"  : str(self.ref),
            "parent": nukeSolitaryList([sub_elem.to_dax() for sub_elem in self.parent])
        }


@attrs.define
class AdagContent:
    xmlns: str = attrs.field(converter=str)
    xsi: str = attrs.field(converter=str)
    version: str = attrs.field(converter=str)
    schemaLocation: str = attrs.field(converter=str)
    job: List[JobContent] = attrs.Factory(list)
    child: List[ChildContent] = attrs.Factory(list)

    @classmethod
    def from_dax(cls, dax_elem):
        return cls(dax_elem["@xmlns"], dax_elem["@xmlns:xsi"], dax_elem["@version"], dax_elem["@xsi:schemaLocation"], [JobContent.from_dax(sub_elem) for sub_elem in forceSequence(dax_elem["job"])],
                   [ChildContent.from_dax(sub_elem) for sub_elem in forceSequence(dax_elem["child"])])

    def to_dax(self):
        return {
            "@xmlns"             : str(self.xmlns),
            "@xmlns:xsi"         : str(self.xsi),
            "@version"           : str(self.version),
            "@xsi:schemaLocation": str(self.schemaLocation),
            "job"                : nukeSolitaryList([sub_elem.to_dax() for sub_elem in self.job]),
            "child"              : nukeSolitaryList([sub_elem.to_dax() for sub_elem in self.child])
        }


@attrs.define
class DaxContent:
    adag: AdagContent

    @classmethod
    def from_dax(cls, dax_elem):
        return cls(AdagContent.from_dax(dax_elem["adag"]))

    def to_dax(self):
        return {"adag": self.adag.to_dax()}


@click.command()
@click.argument("daxInput", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))  # , help="File location of dax input")
@click.argument("daxOutput", type=click.Path(exists=False, writable=True, path_type=pathlib.Path))  # , help="File or directory location to save output at")
@click.option("-p", "--prototypes", help="File location of existing prototypes to be used, omit to generate new ones", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
def main(daxinput: pathlib.Path, daxoutput: pathlib.Path, prototypes: str = None):
    log.setLevel(logging.INFO)
    log.info("running")
    log.info([daxinput, daxoutput, prototypes])
    # parse given .dax
    content = parse_dax(daxinput)

    writeContent = content.to_dax()
    log.info(writeContent)
    log.info(writeContent == xmltodict.parse(daxinput.read_text()))
    log.info(DeepDiff(writeContent, xmltodict.parse(daxinput.read_text())))
    # make prototypes for new tasks or read given
    exit(0)
    #
    dump_dax(daxoutput, content)
    # parse new instances to .dax
    parse_instances()


def parse_dax(fileLoc: pathlib.Path):
    readContent = xmltodict.parse(fileLoc.read_text(), xml_attribs=True)
    convContent = DaxContent.from_dax(readContent)
    return convContent


def dump_dax(daxoutput: pathlib.Path, content: DaxContent):
    writeContent = content.to_dax()
    daxoutput.write_text(xmltodict.unparse(writeContent))


def parse_instances():
    logging.info("parsing instances")
    for new_instance_json_fname in glob.glob("*.json", root_dir="newInstances"):
        wfname = re.fullmatch(r"(.*?)_new\.json", new_instance_json_fname).group(1)
        new_instance_json_fpath_path = pathlib.Path("newInstances").joinpath(new_instance_json_fname)
        logging.info(f"parsing {new_instance_json_fname} to .dax")
        with open(new_instance_json_fpath_path, "rb") as f:
            jsonContent = json.load(f)
        tasksContent: List[TaskContent] = [DataClassUnpack.instantiate(TaskContent, t) for t in
                                           jsonContent["workflow"]["tasks"]]
        output = {
            "adag": {
                "@xmlns"             : "http://pegasus.isi.edu/schema/DAX",
                "@xmlns:xsi"         : "http://www.w3.org/2001/XMLSchema-instance",
                "@version"           : "2.1",
                "@xsi:schemaLocation": "http://pegasus.isi.edu/schema/DAX http://pegasus.isi.edu/schema/dax-2.1.xsd",
                "job"                : [],
                "child"              : [],
            }
        }
        # name <-> id LUT
        nameIDLUT = {}
        for t in tasksContent:
            assert t.name not in nameIDLUT, "task names must be unique"
            assert t.id not in nameIDLUT, "task ids must be unique"
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
            xml = {'@id'         : t.id,
                   '@name'       : t.name,
                   '@namespace'  : wfname,
                   '@numcores'   : t.cores,
                   '@runtime'    : t.runtimeInSeconds * t.cores,
                   '@runtime_raw': t.runtimeInSeconds,
                   'uses'        : []
                   }
            idXmlLUT[t.id] = xml
            idTaskContentLUT[t.id] = t
            output["adag"]["job"].append(xml)
        # create job dag
        for t in tasksContent:
            if len(t.parents) > 0:
                output["adag"]["child"].append({
                    "@ref"  : t.id,
                    "parent": [{
                        "@ref": nameIDLUT[p]
                    } for p in t.parents]
                })
        # create file dependencies
        # todo expand this to also use the files file in wfname_big
        with open(pathlib.Path("results").joinpath(wfname).joinpath(f"{wfname}_files.json"), "rb") as f:
            filesContent: List[dict] = json.load(f)
        filesContent: List[FileContent] = [DataClassUnpack.instantiate(FileContent, f) for f in filesContent]
        # remove tags from task names in filesContent | ie task_name (tag) -> task_name
        # Todo: synthetic tasks will be assigned a random file list from a task with matching name (by choice over available task ids)
        for f in filesContent:
            f.taskName = re.fullmatch(r"(.*?)( \(.*\))?", f.taskName).group(1)

        # collect all files associated with each task name
        # task names in files.json may mismatch against instance.json task names
        baseTaskNameFilesLUT: Dict[str, Dict[str, List[FileContent]]] = {}
        for f in filesContent:
            baseTaskNameFilesLUT.setdefault(f.taskName, {})
            baseTaskNameFilesLUT[f.taskName].setdefault(f.id, [])
            baseTaskNameFilesLUT[f.taskName][f.id].append(f)

        # check if we're parsing a synthetic instance
        # Todo: just checking the first should be enough but is technically risky
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
                shouldFiles: List[FileContent] = pick_random_key_in_dict(baseTaskNameFilesLUT[baseName])
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
                            for bf in pick_random_key_in_dict(baseTaskNameFilesLUT[
                                                                  re.fullmatch(r"(.*?)_(\d{8})", p).group(
                                                                      1)]):  # todo FIX THIS ITS WRONG
                                if not bf.output:
                                    continue
                                if f.fileName == bf.fileName:
                                    parentFile = True
                                    # use the parents output as tagged by their id as input here
                                    taskIdFiles[doNext.id].append(
                                        FileContent(doNext.name, f"{f.fileName}_{nameIDLUT[p]}", f.tag, f.size,
                                                    f.output,
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
        # print(xmltodict.unparse(output, pretty=True))
        pathlib.Path("newInstances").joinpath(f"{wfname}_new.dax").write_text(xmltodict.unparse(output, pretty=True))
        logging.info("done")


def pick_random_key_in_dict(_dict: Dict[any, any]):
    keys = list(_dict.keys())
    return _dict[random.choice(keys)]


if __name__ == '__main__':
    main()
