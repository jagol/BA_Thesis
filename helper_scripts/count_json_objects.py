import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="give path to input file", )
    args = parser.parse_args()
    if args.file:
        with open(args.file, 'r', encoding='utf8') as f:
            d = json.load(f)
            if isinstance(d, dict):
                print(len(d.keys()))
            elif isinstance(d, list):
                print(len(d))
            else:
                print('Format not known!')

if __name__ == '__main__':
    main()
