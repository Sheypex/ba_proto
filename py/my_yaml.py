import atexit

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


# FUNCTIONS
def yaml_load(stream):
    return yaml.load(stream, Loader=Loader)


def yaml_dump(data, stream=None):
    if stream is None:
        return yaml.dump(data, Dumper=Dumper)
    else:
        yaml.dump(data, stream, Dumper=Dumper)


# CONSTANTS
pgconfig_file = "pgconfig.yaml"
pgconfig = yaml_load(open(pgconfig_file, "r"))

# BOOK KEEPING

yaml_dump(pgconfig, open(f".{pgconfig_file}~", "w"))  # make a backup of pgconfig before further execution
atexit.register(lambda: yaml_dump(pgconfig, open(pgconfig_file, "w")))  # save changes to pgconfig permanently
