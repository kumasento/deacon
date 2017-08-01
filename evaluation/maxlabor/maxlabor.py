#!/usr/bin/env python
from __future__ import print_function

import os
import json
import itertools
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


def build(name, config):
    design = config[name]

    param_values = design['params'].values()
    param_names = design['params'].keys()
    args_list = []
    for param_value in itertools.product(*param_values):
        args = ['{0}={1}'.format(x, y) for x, y in zip(param_names, param_value)]
        args_list.append(' '.join(args))

    os.chdir('../build/{0}'.format(name))
    for args in args_list:
        cmd = 'make build {0}'.format(args)
        print('>> ' + cmd)
        os.system(cmd)


def main():
    args = parse_args()
    config = load_config(args.config_file)
    if args.name not in config:
        raise ValueError('{0} is not a valid design name'.format(args.name)) 

    if args.task == 'build':
        build(args.name, config)
    else:
        raise ValueError('{0} is not a valid task'.format(args.task))

if __name__ == '__main__':
    main()
