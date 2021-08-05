import dataclasses
import json
from dataclasses import dataclass
from pprint import pprint
from typing import Dict

import data_types
from db_actions import db_actions
from my_yaml import yaml_load, yaml_dump

# CONSTANTS
nodeConfigsYamlFile = "../cloud/aws/node_configs.yaml"
iperfVcpus = 1
vcpuLimit = 32 - iperfVcpus


def terraID(str):
    return [str[0].upper(), str.split('.')[0], str.split('.')[1]]


# dataclasses
@dataclass()
class instanceSpecs():
    name: str
    ram: int
    vcpus: int
    storageType: data_types.storage_type_enum = data_types.storage_type_enum.ebs.name
    hasGpu: bool = False
    diskType: data_types.disk_type_enum = data_types.disk_type_enum.ssd.name

    def terraID(self):
        return [self.name[0].upper(), self.name.split('.')[0], self.name.split('.')[1]]


@dataclass()
class instanceType():
    arch: data_types.arch_type_enum
    size: Dict[str, instanceSpecs]


def makeNewInstanceType(arch, vcpus, ram, name, sizes=None, asdict=False,
                        storageType=data_types.storage_type_enum.ebs.name,
                        diskType=data_types.disk_type_enum.ssd.name):
    if sizes is None:
        sizes = ['large', 'xlarge', '2xlarge']
    assert len(ram) == len(vcpus)
    assert len(ram) == len(sizes)
    assert isinstance(name, str)
    assert len(name) > 0
    names = [''.join(x) for x in zip([name] * len(sizes), ['.'] * len(sizes), sizes)]
    iType = instanceType(arch, {
        size: instanceSpecs(**specs) for (size, specs) in
        zip(sizes, [{'name': name, 'ram': ram, 'vcpus': vcpus, 'storageType': storageType, 'diskType': diskType} for
                    (ram, vcpus, name) in zip(ram, vcpus, names)])
    })
    return dataclasses.asdict(iType) if asdict else iType


# Constants:
instanceFamilies = {
    'M': {
        'm5': makeNewInstanceType('x86', [2, 4, 8], [8, 16, 32], 'm5'),
        'm5a': makeNewInstanceType('x86', [2, 4, 8], [8, 16, 32], 'm5a'),
        'm5zn': makeNewInstanceType('x86', [2, 4, 8], [8, 16, 32], 'm5zn'),
        'm6g': makeNewInstanceType('arm', [2, 4, 8], [8, 16, 32], 'm6g')
    },
    'C': {
        'c5': makeNewInstanceType('x86', [2, 4, 8], [4, 8, 16], 'c5'),
        'c5a': makeNewInstanceType('x86', [2, 4, 8], [4, 8, 16], 'c5a'),
        'c6g': makeNewInstanceType('arm', [2, 4, 8], [4, 8, 16], 'c6g')
    },
    'R': {
        'r5': makeNewInstanceType('x86', [2, 4, 8], [16, 32, 64], 'r5'),
        'r5a': makeNewInstanceType('x86', [2, 4, 8], [16, 32, 64], 'r5a'),
        'r6g': makeNewInstanceType('arm', [2, 4, 8], [16, 32, 64], 'r6g')
    },
    'D': {
        'd3': makeNewInstanceType('x86', [4, 8], [32, 64], 'd3', ['xlarge', '2xlarge'], storageType='physical',
                                  diskType='hdd')
    },
    'I': {
        'i3': makeNewInstanceType('x86', [2, 4, 8], [15.25, 30.5, 61], 'i3', storageType='physical', diskType='ssd')
    },
    'Z': {
        'z1d': makeNewInstanceType('x86', [2, 4, 8], [16, 32, 64], 'z1d', storageType='physical', diskType='ssd')
    }
}
instanceFamilies = {  # no one has physical storage, its all ebs/ssd
    'M': {
        'm5': makeNewInstanceType('x86', [2, 4, 8], [8, 16, 32], 'm5'),
        'm5a': makeNewInstanceType('x86', [2, 4, 8], [8, 16, 32], 'm5a'),
        'm5zn': makeNewInstanceType('x86', [2, 4, 8], [8, 16, 32], 'm5zn'),
    },
    'C': {
        'c5': makeNewInstanceType('x86', [4, 8], [8, 16], 'c5', ['xlarge', '2xlarge']), # remove c5.large and c5a.large (RAM too low)
        'c5a': makeNewInstanceType('x86', [4, 8], [8, 16], 'c5a', ['xlarge', '2xlarge']), # remove c5.large and c5a.large (RAM too low)
    },
    'R': {
        'r5': makeNewInstanceType('x86', [2, 4, 8], [16, 32, 64], 'r5'),
        'r5a': makeNewInstanceType('x86', [2, 4, 8], [16, 32, 64], 'r5a'),
    },
    'D': {
        'd3': makeNewInstanceType('x86', [4, 8], [32, 64], 'd3', ['xlarge', '2xlarge'])
    },
    'I': {
        'i3': makeNewInstanceType('x86', [2, 4, 8], [15.25, 30.5, 61], 'i3')
    },
    'Z': {
        'z1d': makeNewInstanceType('x86', [2, 4, 8], [16, 32, 64], 'z1d')
    }
}
###
instanceTypes = []
for family, instanceFamily in instanceFamilies.items():
    for instanceTypeBaseName, instanceTypeInfo in instanceFamily.items():
        for size, instanceSpec in instanceTypeInfo.size.items():
            instanceTypes.append(instanceSpec)


# for iT in instanceTypes:
#     print(iT)


def main():
    # load node configs
    nodeConfigs = yaml_load(open(nodeConfigsYamlFile, "r"))
    nodeConfigsList = []
    instanceFamilies = {}
    for family, instanceTypes in nodeConfigs.items():
        instanceFamilies[family] = {}
        for instanceTypeBaseName, instanceTypeInfo in instanceTypes.items():
            instanceFamilies[family][instanceTypeBaseName] = instanceType(instanceTypeInfo['arch'], {})
            for instanceTypeSize, instanceTypeSpecs in instanceTypeInfo['size'].items():
                instanceFamilies[family][instanceTypeBaseName].size[instanceTypeSize] = instanceSpecs(
                    instanceTypeSpecs['name'],
                    instanceTypeSpecs['ram'],
                    instanceTypeSpecs['vcpus'],
                    instanceTypeSpecs['storage_type'],
                    instanceTypeSpecs['gpu'],
                    instanceTypeSpecs['disk_type'])
    # pprint(instanceFamilies)
    ###
    instanceFamilies = {  # no one has physical storage, its all ebs/ssd
        'M': {
            'm5': makeNewInstanceType('x86', [2, 4, 8], [8, 16, 32], 'm5'),
            'm5a': makeNewInstanceType('x86', [2, 4, 8], [8, 16, 32], 'm5a'),
            'm5zn': makeNewInstanceType('x86', [2, 4, 8], [8, 16, 32], 'm5zn'),
            'm6g': makeNewInstanceType('arm', [2, 4, 8], [8, 16, 32], 'm6g')
        },
        'C': {
            'c5': makeNewInstanceType('x86', [4, 8], [8, 16], 'c5', ['xlarge', '2xlarge']),
            # remove c5.large and c5a.large (RAM too low)
            'c5a': makeNewInstanceType('x86', [4, 8], [8, 16], 'c5a', ['xlarge', '2xlarge']),
            # remove c5.large and c5a.large (RAM too low)
            'c6g': makeNewInstanceType('arm', [2, 4, 8], [4, 8, 16], 'c6g')
        },
        'R': {
            'r5': makeNewInstanceType('x86', [2, 4, 8], [16, 32, 64], 'r5'),
            'r5a': makeNewInstanceType('x86', [2, 4, 8], [16, 32, 64], 'r5a'),
            'r6g': makeNewInstanceType('arm', [2, 4, 8], [16, 32, 64], 'r6g')
        },
        'D': {
            'd3': makeNewInstanceType('x86', [4, 8], [32, 64], 'd3', ['xlarge', '2xlarge'])
        },
        'I': {
            'i3': makeNewInstanceType('x86', [2, 4, 8], [15.25, 30.5, 61], 'i3')
        },
        'Z': {
            'z1d': makeNewInstanceType('x86', [2, 4, 8], [16, 32, 64], 'z1d')
        }
    }
    ###
    instanceTypes = []
    for family, instanceFamily in instanceFamilies.items():
        for instanceTypeBaseName, instanceTypeInfo in instanceFamily.items():
            for size, instanceSpec in instanceTypeInfo.size.items():
                instanceTypes.append(instanceSpec)
    noArm = []
    for family, instanceFamily in instanceFamilies.items():
        for instanceTypeBaseName, instanceTypeInfo in instanceFamily.items():
            if instanceTypeInfo.arch != "arm":
                for size, instanceSpec in instanceTypeInfo.size.items():
                    noArm.append(instanceSpec)
    instanceTypes = noArm
    print(len(instanceTypes))
    # pprint(instanceTypes)
    # pprint(sum([x.vcpus for x in instanceTypes]))
    ###
    remaining = [x for x in instanceTypes]
    remaining.sort(key=lambda x: -x.vcpus)
    groupings = []
    while len(remaining) > 0:
        vcpuCount = 0
        group = []
        for spec in remaining:
            if vcpuCount + spec.vcpus <= vcpuLimit:
                group.append(spec)
                vcpuCount += spec.vcpus
        groupings.append(group)
        for x in group:
            remaining.remove(x)
    # pprint(groupings)
    # pprint([[y.vcpus for y in x] for x in groupings])
    # pprint([len([y.vcpus for y in x]) for x in groupings])
    # pprint([sum([y.vcpus for y in x]) for x in groupings])
    # pprint(sum([sum([y.vcpus for y in x]) for x in groupings]))
    ###
    terraParams = [str([y.terraID() for y in x]) for x in groupings]
    terraParams = [x.replace("'", '"') for x in terraParams]
    commands = [f"terraform apply -var 'instances_to_launch={x}' -auto-approve" for x in terraParams]
    for c in commands:
        print(c)
    ###
    # compare = {}
    # for family, instanceTypes in instanceFamilies.items():
    #     compare[family] = {}
    #     for instanceTypeBaseName, instanceTypeInfo in instanceTypes.items():
    #         compare[family][instanceTypeBaseName] = dataclasses.asdict(instanceTypeInfo)
    # pprint(compare)
    ###
    if False:
        # missing
        missing = json.load(open('missing.json', 'r'))
        for testName, instances in missing.items():
            renamed = []
            for instanceName in instances:
                spec = [x for x in instanceTypes if x.name == instanceName][0]
                renamed.append(spec)
            missing[testName] = renamed
        # group
        for testName, instances in missing.items():
            print(f"{testName}:")
            remaining = [x for x in instances]
            remaining.sort(key=lambda x: -x.vcpus)
            groupings = []
            while len(remaining) > 0:
                vcpuCount = 0
                group = []
                for spec in remaining:
                    if vcpuCount + spec.vcpus <= vcpuLimit:
                        group.append(spec)
                        vcpuCount += spec.vcpus
                groupings.append(group)
                for x in group:
                    remaining.remove(x)
            # pprint(groupings)
            # pprint([[y.vcpus for y in x] for x in groupings])
            # pprint([len([y.vcpus for y in x]) for x in groupings])
            # pprint([sum([y.vcpus for y in x]) for x in groupings])
            terraParams = [str([y.terraID() for y in x]) for x in groupings]
            terraParams = [x.replace("'", '"') for x in terraParams]
            commands = [f"terraform apply -var 'instances_to_launch={x}'" for x in terraParams]
            for c in commands:
                print(c)
    ###
    if False:
        for iT in instanceTypes:
            db_actions.insert(
                data_types.node_configs_entry(iT.name, iT.vcpus, iT.ram, iT.hasGpu, iT.storageType, iT.diskType),
                data_types.db_tables_enum.nodeConfigs)
    ###
    if False:
        nodeConfigIdLookup = {}
        ret = db_actions.select('id, "instanceType"', data_types.db_tables_enum.nodeConfigs)
        for i in ret:
            id, name = i
            nodeConfigIdLookup[name] = id
        pprint(nodeConfigIdLookup)
        yaml_dump(nodeConfigIdLookup, open('nodeConfigIdLookup.yaml', 'w'))


if __name__ == '__main__':
    main()
