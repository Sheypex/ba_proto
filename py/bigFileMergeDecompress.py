import os
import re
import subprocess
from pathlib import Path
from typing import Union

import click
import rich.console
import rich

MAXSIZE = 100_000_000  # 100MB in bytes
rc = rich.console.Console()


def decompress(p: Union[str, Path]):
    if isinstance(p, str):
        p = Path(p)
    cmd = ["tar", "-O", "-xf", p.as_posix()]
    rc.log(" ".join(cmd))
    if True or click.confirm(f"do decompress ${' '.join(cmd)}?"):
        unpackTo = p.parent.joinpath(re.match("(.*)\.tar$", p.name).group(1)).as_posix()
        click.confirm(f"outputting to {unpackTo}")
        with open(unpackTo, "bw") as out:
            subprocess.run(cmd, stdout=out)


def merge(*p: Union[str, Path]):
    files = list(p)
    for i, f in enumerate(files):
        if isinstance(f, str):
            files[i] = Path(f)
        else:
            files[i] = f
    files.sort(key=lambda x: x.name)
    cmd = ["cat", *[f.as_posix() for f in files]]
    tarName = re.match("(.*)\.part\d\d$", files[0].as_posix()).group(1)
    rc.log(" ".join(cmd) + f" > {tarName}")
    if True or click.confirm(f"do merge ${' '.join(cmd)} > {tarName}?"):
        with open(tarName, "bw") as out:
            subprocess.run(cmd, stdout=out)


parts = dict()


def processFile(p: Union[str, Path], doTar):
    if isinstance(p, str):
        p = Path(p)
    if doTar:
        if re.match(".*\.tar$", p.name):
            decompress(p)
    else:  # do parts
        m = re.match("(.*)\.part\d\d$", p.name)
        if m:
            parts[m.group(1)] = parts.get(m.group(1), []) + [p]


def checkDir(dir: Union[str, Path], doTar):
    if isinstance(dir, str):
        dir = Path(dir)
    assert dir.is_dir()
    for f in os.listdir(dir):
        pF = dir.joinpath(f)
        if pF.is_dir():
            checkDir(pF, doTar)
        else:
            processFile(pF, doTar)


@click.command()
def cli():
    rootDir = os.getcwd()
    checkDir(Path(rootDir), False)
    for p in parts.values():
        merge(*p)
    checkDir(Path(rootDir), True)


if __name__ == "__main__":
    cli()
