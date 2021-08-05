from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Union

import my_yaml


class AutoNameEnum(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


# this class is literally only used for autocompletion
class db_tables_enum(AutoNameEnum):
    nodeConfigs = auto()
    nodeBenchmarks = auto()
    taskRuntimes = auto()
    initialRuntimes = auto()
    workflows = auto()


# this generates the enum for runtime use
# the value of the enum entries will be equal to the table names found in the yaml
# db_tables_enum = Enum("db_tables", zip(my_yaml.pgconfig["pg_db_tables"], my_yaml.pgconfig["pg_db_tables"]))


# this represents the internal enum type of the same name in the db
class node_component_enum(AutoNameEnum):
    cpu = auto()
    ram = auto()
    memory = auto()
    internet = auto()
    gpu = auto()


# node_component_enum = Enum("node_component_enum", zip(['cpu', 'ram', 'memory', 'internet', 'gpu'], ['cpu', 'ram', 'memory', 'internet', 'gpu']))


class storage_type_enum(AutoNameEnum):
    ebs = auto()
    physical = auto()


# storage_type_enum = Enum("storage_type_enum", zip(["physical", "ebs"], ["physical", "ebs"]))


class disk_type_enum(AutoNameEnum):
    ssd = auto()
    hdd = auto()


# disk_type_enum = Enum("disk_type_enum", zip(["ssd", "hdd"], ["ssd", "hdd"]))


class arch_type_enum(AutoNameEnum):
    arch = auto()
    x86 = auto()


# arch_type_enum = Enum("arch_type_enum", zip(["arch", "x86"], ["arch", "x86"]))


@dataclass()
class db_table_entry:
    pass


@dataclass()
class node_configs_entry(db_table_entry):
    instanceType: str
    vcpus: int = None
    ram: int = None
    hasGpu: bool = None
    storageType: storage_type_enum = storage_type_enum.ebs
    diskType: disk_type_enum = disk_type_enum.ssd


@dataclass()
class node_benchmarks_entry(db_table_entry):
    nodeConfig: Union[
        node_configs_entry, int] = None  # either pass the data of a node_configs_entry or its corresponding id (int)
    benchmarkedComponent: node_component_enum = None
    benchmarkType: str = None
    result: str = None
    benchmarkName: str = None
    units: str = None
    score: float = None


@dataclass()
class node_benchmark_transposed_rankings_entry(db_table_entry):
    nodeConfig: int
    build_linux_kernel1: int
    fio2: int
    fio3: int
    fio4: int
    fio5: int
    fio6: int
    fio7: int
    fio8: int
    fio9: int
    iperf10: int
    iperf11: int
    iperf12: int
    iperf13: int
    john_the_ripper14: int
    john_the_ripper15: int
    ramspeed16: int
    ramspeed17: int
    ramspeed18: int
    ramspeed19: int
    ramspeed20: int
    ramspeed21: int
    ramspeed22: int
    ramspeed23: int
    ramspeed24: int
    ramspeed25: int
    stream26: int
    stream27: int
    stream28: int
    stream29: int


@dataclass()
class workflows_entry(db_table_entry):
    name: str = None


@dataclass()
class task_runtimes_entry(db_table_entry):
    workflow: Union[
        workflows_entry, int] = None  # either pass the data of a workflows_entry or its corresponding id (int)
    task_id: int = None
    node_config: Union[
        node_configs_entry, int] = None  # either pass the data of a node_configs_entry or its corresponding id (int)
    runtime: float = None
