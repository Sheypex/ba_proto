import functools
import itertools
import json
import logging
import pathlib

try:
    from Methylseq_recipes import MethylseqRecipe as testrecp
except ModuleNotFoundError:
    pass
import rich.prompt
import wfcommons
from wfcommons.wfchef.chef import main as chefGo
from wfcommons import WorkflowGenerator


def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("running")
    branch = 2
    match branch:
        case 0:
            parser = wfcommons.wfinstances.NextflowLogsParser(execution_dir=pathlib.Path("jsonToDax/pipeline_big_info"))
            basename = "methylseq"
            filename = basename + "_0"
            for n in itertools.count(1):
                if pathlib.Path(filename + ".json").exists():
                    filename = basename + "_" + str(n)
                else:
                    filename = filename + ".json"
                    break
            if rich.prompt.Confirm.ask(f"Proceed with {filename}?"):
                wf = parser.build_workflow("methylseq")
                wf.write_json(pathlib.Path(filename))
                logging.info("done")
            else:
                logging.info("aborting")
        case 1:
            inst = wfcommons.Instance(input_instance=pathlib.Path("jsonToDax/methylseq_big.json"))
            inst.write_dot(pathlib.Path("./test.dot"))
        case 2:
            chefGo()
        case 3:
            for fn in pathlib.Path("Methyl").iterdir():
                with open(fn, "r") as f:
                    content = json.load(f)
                    content["workflow"]["machines"] = [
                        {
                            "nodeName": "dummy",
                            "cpu": {
                                "count": 1
                            }
                        }
                    ]
                    basenames = {}
                    for t in content["workflow"]["tasks"]:
                        t["name"] = t["name"].replace("_", "-")
                        t["category"] = t["category"].replace("_", "-")
                        if t["category"] in t["name"] and len(t["name"]) > len(t["category"]):
                            basenames[t["name"]] = t["name"].replace(t["category"] + "-", t["category"] + "_")
                            t["name"] = t["name"].replace(t["category"] + "-", t["category"] + "_")
                    for t in content["workflow"]["tasks"]:
                        t["parents"] = [n.replace("_", "-") for n in t["parents"]]
                        t["parents"] = [basenames[n] if n in basenames.keys() else n for n in t["parents"]]
                        t["children"] = [n.replace("_", "-") for n in t["children"]]
                        t["children"] = [basenames[n] if n in basenames.keys() else n for n in t["children"]]
                with open(fn, "w") as f:
                    json.dump(content, f)
        case 4:
            generator = WorkflowGenerator(testrecp.from_num_tasks(250))
            workflow = generator.build_workflow()
            workflow.write_json(pathlib.Path('jsonToDax/methyl.json'))
        case _:
            print("wrong selection")


if __name__ == "__main__":
    main()
