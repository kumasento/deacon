#!/usr/bin/env python

import os
import sys
import multiprocessing as mp
from subprocess import call
from termcolor import colored, cprint 

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
    print '%s for MaxDeep' % colored('Design Space Exploration', 'cyan', attrs=['blink', 'reverse'])

    processes = [
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 1, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 2, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 2, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 4, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 4, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 3, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 3, 3), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 6, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 6, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 6, 3), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 8, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 8, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 8, 4), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 10, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 10, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 12, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 12, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 12, 3), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(100, 12, 4), ))
    ]

    idx = 0
    num_pipes = 4
    while idx < len(processes):
        ps = processes[idx:idx+num_pipes]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        idx += num_pipes

