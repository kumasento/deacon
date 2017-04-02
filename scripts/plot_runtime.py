#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.optimize import curve_fit

class RuntimeDataRow(object):
    def __init__(self, params):
        self.P = int(params['P'])
        self.M = int(params['M'])

        self.runtime = {}
        self.runtime['100MHz'] = self.parse_float(params['Time_100MHz'])
        self.runtime['120MHz'] = self.parse_float(params['Time_120MHz'])
        self.runtime['140MHz'] = self.parse_float(params['Time_140MHz'])

        self.power = {}
        self.power['100MHz'] = self.parse_float(params['Power_100MHz'])
        self.power['120MHz'] = self.parse_float(params['Power_120MHz'])
        self.power['140MHz'] = self.parse_float(params['Power_140MHz'])

    def parse_float(self, param):
        if param.strip() is '':
            return None
        return float(param)

class RuntimeData(object):
    def __init__(self):
        self.datarows = []

        with open('runtime.csv', 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                self.datarows.append(RuntimeDataRow(row))

    def group_by_freq(self):
        P_list = []
        runtime_list = []
        power_list = []

        for idx, freq in enumerate(['100MHz', '120MHz', '140MHz']):
            runtime_list.append([])
            power_list.append([])
            for datarow in self.datarows:
                if datarow.M is not 1:
                    continue
                if datarow.runtime[freq] is None:
                    continue
                if idx is 0:
                    P_list.append(datarow.P)
                runtime_list[idx].append(datarow.runtime[freq])
                power_list[idx].append(datarow.power[freq])

        return P_list, runtime_list, power_list

    def plot_groups_by_freq(self):
        P_list, runtime_list, power_list = self.group_by_freq()
        P_list = np.array(P_list)

        def f(x, a, b):
            return a * 1. / x + b

        popt, pcov = curve_fit(f, P_list, runtime_list[0])

        width = 0.15
        fig, (ax0, ax1) = plt.subplots(1, 2)
        rects0 = ax0.bar(P_list, runtime_list[0], width, color='r')
        rects1 = ax0.bar(P_list+width, runtime_list[1], width, color='g')
        rects2 = ax0.bar(P_list+width*2, runtime_list[2], width, color='b')

        rects3 = ax1.bar(P_list, power_list[0], width, color='r')
        rects4 = ax1.bar(P_list+width, power_list[1], width, color='g')
        rects5 = ax1.bar(P_list+width*2, power_list[2], width, color='b')

        fit, = ax0.plot(P_list, f(P_list, *popt), color='k', linestyle='--', linewidth=2)

        ax0.set_title('Time per Frame among Different $P^{conv}$')
        ax0.set_xlabel('$P^{conv}$')
        ax0.set_xticks(P_list+width)
        ax0.set_xticklabels(P_list)
        ax0.set_xlim(left=0)
        ax0.set_ylabel('Time per frame (s)')
        ax0.legend((rects0, rects1, rects2, fit), ('100MHz', '120MHz', '140MHz', 'fit curve'))

        ax1.set_title('Power Consumption among Different $P^{conv}$')
        ax1.set_ylim([10, None])
        ax1.set_ylabel('Power (W)')
        ax1.legend((rects3, rects4, rects5), ('100MHz', '120MHz', '140MHz'))

        fig.set_figheight(4)
        fig.set_figwidth(12)
        fig.savefig('runtime_power_by_freq.pdf')


if __name__ == '__main__':
    runtime_data = RuntimeData()
    runtime_data.plot_groups_by_freq()

