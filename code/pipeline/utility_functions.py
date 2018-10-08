import json
import argparse
from typing import Tuple, Dict


corpus_config = Dict[str, Dict[str, str]]


def get_corpus_config(unit: str) -> Tuple[str, corpus_config]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--server',
        help='indicate if local paths or server paths should be used',
        action='store_true')
    parser.add_argument(
        '-c',
        '--corpus',
        help='indicate name of corpus to be processed: europarl; dblp;'
    )
    args = parser.parse_args()

    with open('configs.json', 'r', encoding='utf8') as f:
        configs = json.load(f)
        if args.server:
            location = 'server'
        else:
            location = 'local'

        dblp = 'dblp'
        euprl = 'europarl'
        sp = 'sp'

        if args.corpus == dblp:
            return dblp, configs[location][dblp][unit]
        elif args.corpus == euprl:
            return euprl, configs[location][euprl][unit]
        elif args.corpus == sp:
            return sp, configs[location][sp][unit]
        else:
            raise Exception('Error! Corpus not known.')
