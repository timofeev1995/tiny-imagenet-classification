import datetime
import pydoc

import yaml


def get_object(class_path: str, *args, **kwargs):
    cls = pydoc.locate(class_path)
    obj = cls(*args, **kwargs)
    return obj


def get_cur_time_str():
    result = datetime.datetime.utcnow().strftime('%Y_%m_%d__%H_%M')
    return result


def load_yaml(path):
    with open(path, 'r') as file:
        _object = yaml.safe_load(file)
    return _object


def dump_yaml(_object, path):
    with open(path, 'w') as file:
        yaml.safe_dump(_object, file)
