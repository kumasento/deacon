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
        ax1.set_xlabel('$P^{conv}$')
        ax1.set_xticks(P_list+width)
        ax1.set_xticklabels(P_list)
        ax1.set_xlim(left=0)
        ax1.set_ylim([10, None])
        ax1.set_ylabel('Power (W)')
        ax1.legend((rects3, rects4, rects5), ('100MHz', '120MHz', '140MHz'))

        fig.set_figheight(4)
        fig.set_figwidth(12)
        fig.savefig('runtime_power_by_freq.pdf')

    def group_by_M(self):
        P_list = []
        runtime_list = [[], []]
        power_list = [[], []]
        energy_list = [[], []]

        for datarow in self.datarows:
            if datarow.M is 1:
                continue
            if datarow.runtime['100MHz'] is None:
                continue

            P_list.append(datarow.P)
            runtime_list[1].append(datarow.runtime['100MHz'])
            power_list[1].append(datarow.power['100MHz'])
            energy_list[1].append(float(datarow.power['100MHz'] * datarow.runtime['100MHz']))

        for datarow in self.datarows:
            if datarow.M is not 1:
                continue
            if datarow.P not in P_list:
                continue
            runtime_list[0].append(datarow.runtime['100MHz'])
            power_list[0].append(datarow.power['100MHz'])
            energy_list[0].append(datarow.power['100MHz'] * datarow.runtime['100MHz'])

        return P_list, runtime_list, power_list, energy_list

    def group_by_M_effective_P(self):
        P_list = []
        keep_P_list = []
        runtime_list = [[], []]
        power_list = [[], []]
        energy_list = [[], []]

        for datarow in self.datarows:
            if datarow.M is 1:
                continue
            if datarow.runtime['100MHz'] is None:
                continue

            P_list.append(datarow.P / datarow.M)
            runtime_list[1].append(datarow.runtime['100MHz'])
            power_list[1].append(datarow.power['100MHz'])
            energy_list[1].append(float(datarow.power['100MHz'] * datarow.runtime['100MHz']))

        for datarow in self.datarows:
            if datarow.M is not 1:
                continue
            if datarow.P not in P_list:
                continue

            keep_P_list.append(datarow.P)
            runtime_list[0].append(datarow.runtime['100MHz'])
            power_list[0].append(datarow.power['100MHz'])
            energy_list[0].append(datarow.power['100MHz'] * datarow.runtime['100MHz'])

        while len(keep_P_list) != len(P_list):
            for idx, P in enumerate(P_list):
                if P not in keep_P_list:
                    del P_list[idx]
                    del runtime_list[1][idx]
                    del power_list[1][idx]
                    del energy_list[1][idx]

        return P_list, runtime_list, power_list, energy_list


    def plot_groups_by_M(self):
        P_list, runtime_list, power_list, energy_list =  self.group_by_M()
        P_list = np.array(P_list)

        eff_P_list, _, _, eff_energy_list =  self.group_by_M_effective_P()
        eff_P_list = np.array(eff_P_list)
        print eff_P_list
        print eff_energy_list

        width = 0.30
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
        rects0 = ax0.bar(P_list, runtime_list[0], width, color='r')
        rects1 = ax0.bar(P_list+width, runtime_list[1], width, color='b')

        rects2 = ax1.bar(P_list, power_list[0], width, color='r')
        rects3 = ax1.bar(P_list+width, power_list[1], width, color='b')

        rects4 = ax2.bar(P_list, energy_list[0], width, color='r')
        rects5 = ax2.bar(P_list+width, energy_list[1], width, color='b')

        rects6 = ax3.bar(eff_P_list, eff_energy_list[0], width, color='r')
        rects7 = ax3.bar(eff_P_list+width, eff_energy_list[1], width, color='b')

        ax0.set_title('Time per Frame among Different $P^{conv}$')
        ax0.set_xlabel('$P^{conv}$')
        ax0.set_xticks(P_list+width)
        ax0.set_xticklabels(P_list)
        ax0.set_xlim(left=0)
        ax0.set_ylabel('Time per frame (s)')
        ax0.legend((rects0, rects1), ('$M^{conv}=1$', '$M^{conv}=2$'))

        ax1.set_title('Power Consumption among Different $P^{conv}$')
        ax1.set_xlabel('$P^{conv}$')
        ax1.set_xticks(P_list+width)
        ax1.set_xticklabels(P_list)
        ax1.set_xlim(left=0)
        ax1.set_ylim([10, None]) 
        ax1.set_ylabel('Power (W)')

        ax2.set_title('Energy per Frame among Different $P^{conv}$')
        ax2.set_xlabel('$P^{conv}$')
        ax2.set_xticks(P_list+width)
        ax2.set_xticklabels(P_list)
        ax2.set_xlim(left=0)
        ax2.set_ylabel('Energy per Frame (J)')
        ax2.legend((rects4, rects5), ('$M^{conv}=1$', '$M^{conv}=2$'))

        ax3.set_title('Energy per Frame among Effective $P^{conv}$')
        ax3.set_xlabel('Effective $P^{conv}$')
        ax3.set_xticks(eff_P_list+width)
        ax3.set_xticklabels(eff_P_list)
        ax3.set_xlim(left=0)
        ax3.set_ylabel('Energy per Frame (J)')
        ax3.legend((rects6, rects7), ('$M^{conv}=1$', '$M^{conv}=2$'))

        fig.set_figheight(10)
        fig.set_figwidth(12)
        fig.savefig('energy_by_M.pdf')

if __name__ == '__main__':
    runtime_data = RuntimeData()
    runtime_data.plot_groups_by_freq()
    runtime_data.plot_groups_by_M()

