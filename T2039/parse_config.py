import json
from pathlib import Path
from collections import OrderedDict

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def json_to_config(args):
    args = args.parse_args()
    cfg_fname = Path(args.config)
    config = read_json(cfg_fname)
    return config