import itertools
import logging
import pathlib

import rich.prompt
import wfcommons


def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("running")
    branch = 2
    match branch:
        case 0:
            parser = wfcommons.wfinstances.NextflowLogsParser(execution_dir=pathlib.Path("tmp/pipeline_info"))
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
            inst = wfcommons.Instance(input_instance=pathlib.Path("methylseq_0.json"))
            inst.write_dot(pathlib.Path("./test.dot"))
        case 2:
            ...
        case _:
            print("wrong selection")


if __name__ == "__main__":
    main()
