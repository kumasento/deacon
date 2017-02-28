#!/usr/bin/env python

import os
import sys
import multiprocessing as mp
from subprocess import call
from termcolor import colored, cprint 

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUILD_DIR = os.path.join(ROOT_DIR, '../MaxDeep/build')

class MaxDeepBuildParam(object):
    def __init__(self, K, N, F, C):
        self.num_pipes            = N
        self.kernel_size          = K
        self.freq                 = F
        self.multi_pumping_factor = C

    def getParams(self):
        return [
            'NUM_PIPES=%d'            % self.num_pipes,
            'KERNEL_SIZE=%d'          % self.kernel_size,
            'FREQ=%d'                 % self.freq,
            'MULTI_PUMPING_FACTOR=%d' % self.multi_pumping_factor
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
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 100, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 100, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 125, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 125, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 150, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 150, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 175, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 175, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 200, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 200, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 225, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 225, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 250, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 250, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 275, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 275, 2), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 300, 1), )),
        mp.Process(target=build, args=(MaxDeepBuildParam(4, 6, 300, 2), ))
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
