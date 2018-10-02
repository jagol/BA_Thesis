import json
import argparse

def get_config(unit):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--server',
        help="indicate if local paths or server paths should be used",
        action='store_true')
    args = parser.parse_args()
    with open('configs.json', 'r', encoding='utf8') as f:
        configs = json.load(f)
        if args.server:
            location = 'server'
        else:
            location = 'local'

    return configs[location][unit]