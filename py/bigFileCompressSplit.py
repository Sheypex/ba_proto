import os
import re
import subprocess
from pathlib import Path
from typing import Union

import click
import rich.console

MAXSIZE = 100_000_000  # 100MB in bytes
rc = rich.console.Console()


def compress(p: Union[str, Path]):
    if isinstance(p, str):
        p = Path(p)
    cmd = ["tar", "--use-compress-program=pigz", "-cf", f"{p.as_posix()}.tar", p.as_posix()]
    rc.log(" ".join(cmd))
    subprocess.run(cmd)


def split(p: Union[str, Path]):
    if isinstance(p, str):
        p = Path(p)
    cmd = ["split", "-d", "-b", "100M", p.as_posix(), f"{p.as_posix()}.part"]
    rc.log(" ".join(cmd))
    subprocess.run(cmd)


def gitAdd(p: Union[str, Path]):
    if isinstance(p, str):
        p = Path(p)
    cmd = ["git", "add", p.as_posix()]
    rc.log(" ".join(cmd))
    subprocess.run(cmd)


def processFile(p: Union[str, Path], doGitAdd, clean):
    if isinstance(p, str):
        p = Path(p)
    if clean:
        m1 = re.match("(.*)\.tar$", p.as_posix())
        if m1:
            if Path(m1.group(1)).exists() or click.confirm(f"Safe to remove {p.as_posix()}?"):
                rc.log(f"Cleaning {p.as_posix()}")
                os.remove(p)
            return
        m2 = re.match("(.*)\.part\d\d$", p.as_posix())
        m3 = re.match("(.*)\.tar\.part\d\d$", p.as_posix())
        if m2:
            if Path(m2.group(1)).exists() or Path(m3.group(1)).exists() or click.confirm(f"Safe to remove {p.as_posix()}?"):
                rc.log(f"Cleaning {p.as_posix()}")
                os.remove(p)
            return
        return
    if not p.stat().st_size > MAXSIZE:
        return
    if re.match("(.*)\.tar$", p.as_posix()):
        return
    if re.match("(.*)\.part\d\d$", p.as_posix()):
        return
    #
    rc.log(f"Found {p.as_posix()} > 100M")
    compress(p)
    tarP = p.parent.joinpath(f"{p.name}.tar")
    if tarP.stat().st_size > MAXSIZE:
        split(tarP)
        if doGitAdd:
            partNum = 0
            partP = tarP.parent.joinpath(f"{tarP.name}.part{partNum:02}")
            while partP.exists():
                gitAdd(partP)
                partNum = partNum + 1
                partP = tarP.parent.joinpath(f"{tarP.name}.part{partNum:02}")

    else:
        if doGitAdd:
            gitAdd(tarP)
    rc.log("")


def checkDir(dir: Union[str, Path], doGitAdd, onlyClean):
    if isinstance(dir, str):
        dir = Path(dir)
    assert dir.is_dir()
    for f in os.listdir(dir):
        pF = dir.joinpath(f)
        if pF.is_dir():
            checkDir(pF, doGitAdd, onlyClean)
        else:
            processFile(pF, doGitAdd, onlyClean)


@click.command()
@click.option("--add/--no-add", default=False)
@click.option("--clean/--no-clean", default=False)
def cli(add, clean):
    rootDir = os.getcwd()
    if clean:
        checkDir(Path(rootDir), add, True)
    else:
        checkDir(Path(rootDir), add, True)
        checkDir(Path(rootDir), add, False)


if __name__ == "__main__":
    cli()
