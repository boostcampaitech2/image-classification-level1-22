from pathlib import Path
from collections import OrderedDict
import json

def read_json(path):
    path = Path(path)
    with open(path, "r") as json_file:
        config = json.load(json_file, object_hook=OrderedDict)
        return config

def json_to_config(args):
    args = args.parse_args()
    config_path = Path(args.config)
    config = read_json(config_path)
    return config