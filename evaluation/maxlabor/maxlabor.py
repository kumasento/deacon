#!/usr/bin/env python
from __future__ import print_function

import csv
import os
import json
import itertools
from argparse import ArgumentParser
from collections import OrderedDict


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', help='name of the design')
    parser.add_argument('-t', '--task', help='name of the task')
    parser.add_argument('--config-file', default='config.json', help='where the designs configuration file is')

    return parser.parse_args()


def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.loads(f.read(), object_pairs_hook=OrderedDict)


def build(name, config):
    design = config[name]

    param_values = design['params'].values()
    param_names = design['params'].keys()
    args_list = []
    for param_value in itertools.product(*param_values):
        args = ['{0}={1}'.format(x, y) for x, y in zip(param_names, param_value)]
        args_list.append(' '.join(args))

    for idx, args in enumerate(args_list):
        build_args = ['{0}={1}'.format(x, y) for x, y in design['buildParams'].items()]
        args += ' ' + ' '.join(build_args)
        args_list[idx] = args

    os.chdir('../build/{0}'.format(name))
    for args in args_list:
        cmd = 'make build {0}'.format(args)
        print('>> ' + cmd)
        os.system(cmd)


def analyse(name, config):
    """
    Analyse and collect data from the builds of design with name = "name"
    :param name:
    :param config:
    :return:
    """
    design = config[name]

    param_value_list = design['params'].values()
    param_names = design['params'].keys()

    for file in design['config']['files']:
        name = file['name']

        with open('../data/{0}.csv'.format(name), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(param_names + ['LUT', 'FF', 'BRAM', 'DSP'])

            for param_values in itertools.product(*param_value_list):
                print('permutation: {0}'.format(param_values))

                path = file['path'].format(*param_values)
                path = os.path.join(design['buildParams']['MAXCOMPILER_BUILD_DIR'], path)
                lineNumber = file['lineNumber']

                with open(path, 'r') as f:
                    usage = f.readlines()[lineNumber - 1].strip().split()[:4]
                csvwriter.writerow(param_values + tuple(usage))



def main():
    args = parse_args()
    config = load_config(args.config_file)
    if args.name not in config:
        raise ValueError('{0} is not a valid design name'.format(args.name)) 

    if args.task == 'build':
        build(args.name, config)
    elif args.task == 'analyse':
        analyse(args.name, config)
    else:
        raise ValueError('{0} is not a valid task'.format(args.task))

if __name__ == '__main__':
    main()
