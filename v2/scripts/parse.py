#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from sklearn import linear_model
import matplotlib
import matplotlib.pyplot as plt

font = {
    'family' : 'monospace',
    'size'   : 22
}

matplotlib.rc('font', **font)

BUILD_ROOT_DIR='/mnt/data/scratch/rz3515/builds/'

class MaxDeepBuild(object):
    def __init__(self, root_dir, param_list):
        self.root_dir          = root_dir
        self.conv_num_pipes    = int(param_list[0])
        self.conv_height       = int(param_list[1])
        self.conv_width        = int(param_list[2])
        self.conv_num_channels = int(param_list[3])
        self.conv_num_filters  = int(param_list[4])
        self.conv_kernel_size  = int(param_list[5])
        self.fc_height         = int(param_list[6])
        self.fc_width          = int(param_list[7])
        self.fc_num_row_pipes  = int(param_list[8])
        self.fc_num_col_pipes  = int(param_list[9])
        self.mpdp              = int(param_list[10])
        self.mpc               = int(param_list[11])
        self.freq              = int(param_list[12][:-3])

    def __str__(self):
        return (
            'conv_num_pipes    : ' + self.conv_num_pipes    + '\n'
            'conv_height       : ' + self.conv_height       + '\n'
            'conv_width        : ' + self.conv_width        + '\n'
            'conv_num_channels : ' + self.conv_num_channels + '\n'
            'conv_num_filters  : ' + self.conv_num_filters  + '\n'
            'conv_kernel_size  : ' + self.conv_kernel_size  + '\n'
            'fc_height         : ' + self.fc_height         + '\n'
            'fc_width          : ' + self.fc_width          + '\n'
            'fc_num_row_pipes  : ' + self.fc_num_row_pipes  + '\n'
            'fc_num_col_pipes  : ' + self.fc_num_col_pipes  + '\n'
            'mpdp              : ' + self.mpdp              + '\n'
            'mpc               : ' + self.mpc               + '\n'
            'freq              : ' + self.freq              + '\n'
        )

    def load_build_log(self):
        build_log_file_name = os.path.join(self.root_dir, '_build.log')
        with open(build_log_file_name, 'r') as f:
            self.build_log_lines = f.readlines()

    def locate_resource_usage_in_build_log(self):
        build_log_num_lines = len(self.build_log_lines)
        idx = build_log_num_lines - 1
        regexp = re.compile('(.*)FINAL RESOURCE USAGE(.*)')
        while idx >= 0 and idx >= build_log_num_lines - 100:
            curr_line = self.build_log_lines[idx]
            if regexp.search(curr_line) is not None:
                return idx
            idx -= 1
        return -1

    def parse_resource_usage_line(self, line):
        line = line.strip()
        regexp = re.compile('(.*)PROGRESS:(\s*)(.*):(\s*)(\d*) / (\d*)(\s*)\((.*)%\)(.*)')
        result = regexp.search(line)

        name       = result.group(3)
        usage      = int(result.group(5))
        total      = int(result.group(6))
        percentage = float(result.group(8))
        return (name, usage, total, percentage)

    def parse_resource_usage_from_build_log(self):
        resource_usage_line_idx = self.locate_resource_usage_in_build_log() 
        if resource_usage_line_idx == -1:
            print 'This build is not yet finished'
            return None

        resource_usage_lines = self.build_log_lines[
                resource_usage_line_idx+1:resource_usage_line_idx + 8]

        resource_usage_dict = {}
        for idx in [1, 2, 3, 5, 6]:
            ( name
            , usage
            , total
            , percentage) = self.parse_resource_usage_line(resource_usage_lines[idx])

            resource_usage_dict[name] = (usage, total, percentage)
        return resource_usage_dict

    def process_build_log(self):
        self.resource_usage = build.parse_resource_usage_from_build_log()
        self.finished = self.resource_usage is not None

class DesignModel(object):
    def __init__(self, builds):
        self.builds = builds

    def get_resource_usage_list(self, name):
        result = []
        for build in self.builds:
            if name == 'LOGIC':
                result.append(
                    build.resource_usage['LUTs'][0] +
                    build.resource_usage['Primary FFs'][0])
            elif name == 'DSP':
                result.append(build.resource_usage['DSP blocks'][0])
            elif name == 'BRAM':
                result.append(build.resource_usage['Block memory (BRAM18)'][0])
        return result

    def get_param_list(self, name):
        result = []
        for build in builds:
            if name == 'P_conv':
                result.append(build.conv_num_pipes)
            elif name == 'M_conv':
                result.append(build.mpc)
        return result

    def group_builds_by_M_conv(self):
        groups = {}
        for idx in range(len(self.builds)):
            M = self.M_conv[idx]
            if groups.has_key(M):
                groups[M].append(idx)
            else:
                groups[M] = [idx]

        return groups

    def build(self):
        self.max_logic = self.builds[0].resource_usage['LUTs'][1]
        self.max_dsp   = self.builds[0].resource_usage['DSP blocks'][1]
        self.max_bram  = self.builds[0].resource_usage['Block memory (BRAM18)'][1]

        self.logic = self.get_resource_usage_list('LOGIC')
        self.dsp   = self.get_resource_usage_list('DSP')
        self.bram  = self.get_resource_usage_list('BRAM')

        self.P_conv = self.get_param_list('P_conv')
        self.M_conv = self.get_param_list('M_conv')

        # plt.subplot(121)

    def train(self):
        self.models = {}
        groups = self.group_builds_by_M_conv()
        for M_conv in groups:
            group = groups[M_conv]
            P_conv_list = np.array(self.P_conv)[group].reshape(len(group), 1)
            dsp_list    = np.array(self.dsp)[group].reshape(len(group), 1)
            logic_list  = np.array(self.logic)[group].reshape(len(group), 1)

            regr = linear_model.LinearRegression()
            self.models[M_conv] = regr.fit(
                    P_conv_list,
                    np.concatenate((logic_list, dsp_list), axis=1))

    def predict(self, P_conv, M_conv):
        return self.models[M_conv].predict([[P_conv]])

    def plot(self):
        colors = ['r', 'g', 'b', 'y']
        markers = ['x', 'o', '^', '>']

        fig = plt.figure(figsize=(24, 7))
        ax0 = fig.add_subplot(1, 3, 1)
        ax1 = fig.add_subplot(1, 3, 2)

        groups = self.group_builds_by_M_conv()
        for M_conv in groups:
            group = groups[M_conv]
            P_conv_list = np.array(self.P_conv)[group].reshape(len(group), 1)
            dsp_list    = np.array(self.dsp)[group].reshape(len(group), 1)
            logic_list  = np.array(self.logic)[group].reshape(len(group), 1)
            
            ax0.scatter(
                P_conv_list,
                logic_list,
                marker=markers[M_conv-1],
                color=colors[M_conv-1],
                label='$M^{conv} = %d$' % (M_conv))
            ax0.plot(
                np.sort(P_conv_list, axis=0),
                self.models[M_conv].predict(np.sort(P_conv_list, axis=0))[:, 0],
                color=colors[M_conv-1])

            ax1.scatter(
                P_conv_list,
                dsp_list,
                marker=markers[M_conv-1],
                color=colors[M_conv-1],
                label='$M^{conv} = %d$' % (M_conv))
            ax1.plot(
                np.sort(P_conv_list, axis=0),
                self.models[M_conv].predict(np.sort(P_conv_list, axis=0))[:, 1],
                color=colors[M_conv-1])
        ax0.set_title('Logic Usage')
        ax0.set_xlabel('$P^{conv}$', fontsize=22)
        ax0.set_xlim(left=0)
        ax0.set_ylabel('$U_{logic}$', fontsize=22)
        # ax0.legend()
        ax1.set_title('DSP Usage')
        ax1.set_xlabel('$P^{conv}$', fontsize=22)
        ax1.set_xlim(left=0)
        ax1.set_ylabel('$U_{dsp}$', fontsize=22)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig('logic_dsp.pdf')

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print 'usage: %s [date]' % sys.argv[0] 

    root_dir = os.path.join(BUILD_ROOT_DIR, sys.argv[1])
    print root_dir
    builds = []
    for build_dir_name in os.listdir(root_dir):
        # print build_dir_name
        m = re.search(
            'MaxDeep_MAX3424A_DFE_'
            'N_(.*)_H_(.*)_W_(.*)_C_(.*)_F_(.*)_K_(.*)_'
            'FH_(.*)_FW_(.*)_FR_(.*)_FC_(.*)_MPDP_(.*)_C_(.*)_F_(.*)', build_dir_name)

        if m is None:
            continue
        build = MaxDeepBuild(os.path.join(root_dir, build_dir_name), m.groups())
        build.load_build_log()
        build.process_build_log()
        if build.finished and build.freq == 100:
            builds.append(build)

    design_model = DesignModel(builds)
    print 'Building and training the design model ...'
    design_model.build()
    design_model.train()

    print 'Plotting the dataset ...'
    design_model.plot()

    print 'Running ILP ...'
    for M_conv in [1, 2, 3]:
        P_conv = 1
        while True:
            U_logic, U_dsp = design_model.predict(P_conv, M_conv)[0]
            if U_logic >= design_model.max_logic or U_dsp >= design_model.max_dsp:
                break
            P_conv += 1
        print P_conv
