#!/usr/bin/env python3
import json
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', help='name of the design')
    parser.add_argument('-t', '--task', help='name of the task')
    parser.add_argument('--config-file', default='config.json', help='where the designs configuration file is')

    return parser.parse_args()


def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.loads(f.read())


def main():
    args = parse_args()

if __name__ == '__main__':
    main()