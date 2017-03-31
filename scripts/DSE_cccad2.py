#!/usr/bin/env python

import os
import sys
import multiprocessing as mp
from subprocess import call
# from termcolor import colored, cprint 

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUILD_DIR = os.path.join(ROOT_DIR, '../MaxDeep/build')

class MaxDeepBuildParam(object):
    def __init__(self, F, N, C, FR=1, FC=1, MPDP=1):
        self.freq             = F
        self.pipe             = N
        self.mpc              = C
        self.mpdp             = MPDP
        self.num_fc_row_pipes = FR
        self.num_fc_col_pipes = FC

    def getParams(self):
        return [
            'FREQ=%d'                 % self.freq,
            'NUM_PIPES=%d'            % self.pipe,
            'MULTI_PUMPING_FACTOR=%d' % self.mpc,
            'NUM_FC_ROW_PIPES=%d'     % self.num_fc_row_pipes,
            'NUM_FC_COL_PIPES=%d'     % self.num_fc_col_pipes,
            'NUM_MPDP_FACTOR=%d'      % self.mpdp
        ]

def build(params):
    print 'Running build ...'
    print 'ROOT_DIR:  %s' % ROOT_DIR
    print 'BUILD_DIR: %s' % DEFAULT_BUILD_DIR
    os.chdir(DEFAULT_BUILD_DIR)
    print 'CURR_DIR:  %s' % os.path.abspath('./')
    print 'Run make'
    call(['make', 'build'] + params.getParams())

if __name__ == '__main__':
    # print '%s for MaxDeep' % colored('Design Space Exploration', 'cyan', attrs=['blink', 'reverse'])

    processes = [
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 24, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 24, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 24, 3), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 30, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 30, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 30, 3), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 36, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 36, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 36, 3), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 48, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 48, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 48, 3), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 56, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 56, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 56, 3), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 72, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 72, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 72, 3), )),
    ]

    idx = 0
    num_pipes = 16
    while idx < len(processes):
        ps = processes[idx:idx+num_pipes]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        idx += num_pipes

