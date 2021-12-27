#!/usr/bin/env python3

import argparse
import math
import os
import random
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import toml
from tqdm import tqdm

MAX_COST = 1e20


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_dsp_usage(PC, PF, cfg: Dict, name: str, board_cfg: Dict):
    """Calculate the DSP usage of a single layer."""
    layer_cfg = cfg["layers"][name]
    layer_type = layer_cfg["TYPE"]
    K = layer_cfg["K"]
    dsp_key = f"b{cfg['global']['BW']}w{cfg['global']['WBW']}"

    if K == 3:
        if layer_type == "STANDARD":
            return board_cfg[dsp_key]["STANDARD_3x3_DSP"] * PC * PF
        if layer_type == "DEPTHWISE_SEPARABLE":
            return (
                board_cfg[dsp_key]["DEPTHWISE_3x3_DSP"] * PC
                + board_cfg[dsp_key]["POINTWISE_DSP"] * PC * PF
            )
        assert False
    assert False


def get_cycles(PC, PF, cfg: Dict, name: str):
    layer_cfg = cfg["layers"][name]
    layer_type = layer_cfg["TYPE"]
    H = layer_cfg["H"]
    W = layer_cfg["W"]
    C = layer_cfg["C"]
    F = layer_cfg["F"]
    S = layer_cfg["S"]

    if layer_type == "STANDARD":
        return H * W * C * F * S * S // PC // PF
    if layer_type == "DEPTHWISE_SEPARABLE":
        return H * W * C * F * S * S // PC // PF
    assert False


def get_non_negative_divisible(N: int) -> List[int]:
    return [x for x in range(1, N) if N % x == 0]


def get_par_binaries_and_sum(
    model: Any, key: str, layer_name: str, layer_cfg: Dict
) -> Tuple[List[Any], Any]:
    ps = []
    pbs = []
    for i in get_non_negative_divisible(layer_cfg[key[1:]]):
        model.add_component(f"{layer_name}_{key}_{i}", pyo.Var(within=pyo.Binary))
        pbs.append(getattr(model, f"{layer_name}_{key}_{i}"))
        ps.append(pbs[-1] * i)
    p = sum(ps)
    return pbs, p


def monte_carlo(cfg, board_cfg):
    candidates = []
    for i, name in enumerate(cfg["layers"]):
        layer_cfg = cfg["layers"][name]
        if i == 0:
            candidates.append(get_non_negative_divisible(layer_cfg["C"]))
        candidates.append(get_non_negative_divisible(layer_cfg["F"]))

    sols = []
    best_val = -1
    best_idx = 0
    for _ in range(100000):
        sol = [np.random.randint(low=0, high=len(cands)) for cands in candidates]
        ps = [candidates[i][j] for i, j in enumerate(sol)]

        dsp_usage = 0
        cycles = []
        for i, name in enumerate(cfg["layers"]):
            layer_cfg = cfg["layers"][name]
            dsp_usage += get_dsp_usage(ps[i], ps[i + 1], cfg, name, board_cfg)
            cycles.append(get_cycles(ps[i], ps[i + 1], cfg, name))

        if dsp_usage > board_cfg["DSP"] * 0.9:
            continue

        # sum_diff = sum([abs(y - x) for x, y in zip(cycles[:-1], cycles[1:])])
        sum_cycles = sum(cycles)
        # sum_diff = sum([abs(x - sum_cycles // len(cycles)) for x in cycles])
        obj = max(cycles)
        sols.append((ps, obj, dsp_usage, cycles))

        if best_val == -1 or best_val >= obj:
            best_val = obj
            best_idx = len(sols) - 1

    print(best_val, sols[best_idx][0], sols[best_idx][2], sols[best_idx][3])


class SimulatedAnnealing:
    def __init__(self, cfg, board_cfg, dsp_scale=0.9, incl_fst=False):
        self.cfg = cfg
        self.board_cfg = board_cfg
        self.dsp_scale = dsp_scale
        self.incl_fst = incl_fst

        self.initial_temp = 90
        self.final_temp = 0.1
        self.alpha = 0.0001
        self.candidates = self.get_candidates()

    def get_candidates(self):
        candidates = []
        for i, name in enumerate(self.cfg["layers"]):
            layer_cfg = self.cfg["layers"][name]
            if i == 0:
                candidates.append(get_non_negative_divisible(layer_cfg["C"]))
            candidates.append(get_non_negative_divisible(layer_cfg["F"]))
        return candidates

    def get_neighbour(self, state):
        return [
            random.choice(
                list(
                    set(
                        [
                            min(state[i] + 1, len(self.candidates[i]) - 1),
                            state[i],
                            max(state[i] - 1, 0),
                        ]
                    )
                )
            )
            for i in range(len(state))
        ]

    def get_ps(self, state):
        return [self.candidates[i][j] for i, j in enumerate(state)]

    def get_cycles(self, state):
        ps = self.get_ps(state)
        cycles = []
        for i, name in enumerate(self.cfg["layers"]):
            cycles.append(get_cycles(ps[i], ps[i + 1], self.cfg, name))
        return cycles

    def get_dsp_usage(self, state):
        ps = self.get_ps(state)
        dsp_usage = 0
        for i, name in enumerate(self.cfg["layers"]):
            dsp_usage += get_dsp_usage(ps[i], ps[i + 1], self.cfg, name, self.board_cfg)
        return dsp_usage

    def get_cost(self, state):
        dsp_usage = self.get_dsp_usage(state)
        cycles = self.get_cycles(state)

        if dsp_usage > self.board_cfg["DSP"] * self.dsp_scale:
            return MAX_COST

        # if not self.incl_fst:
        #     return max(cycles[1:]) - min(cycles[1:]) + np.mean(cycles[1:])

        # return max(cycles) - min(cycles) + np.mean(cycles)
        return max(cycles) * dsp_usage

    def run(self, max_iter):
        curr_temp = self.initial_temp
        sol = np.zeros(len(self.candidates), dtype=int)
        # while self.get_cost(sol) == MAX_COST:
        #     sol = [
        #         np.random.randint(low=0, high=len(cands)) for cands in self.candidates
        #     ]

        costs = [self.get_cost(sol)]

        for i in tqdm(range(max_iter)):
            if curr_temp <= self.final_temp:
                break

            neighbour = self.get_neighbour(sol)
            cost_diff = self.get_cost(neighbour) - self.get_cost(sol)
            if cost_diff < 0:
                sol = neighbour
            else:
                if random.uniform(0, 1) < math.exp(-cost_diff / curr_temp):
                    sol = neighbour
            costs.append(self.get_cost(sol))
            # if i % 10000 == 0:
            #     print(f"i: {i:10d} current cost: {costs[-1]}")
            curr_temp -= self.alpha

        return sol, costs


def main():
    set_seed(42)

    parser = argparse.ArgumentParser("Explore Deacon configuration.")
    parser.add_argument("-f", "--cfg", type=str, help="Model configuration")
    parser.add_argument("-b", "--board", type=str, help="Board configuration")
    parser.add_argument("--dsp-scale", type=float, help="DSP usage scale")
    parser.add_argument(
        "--incl-fst", action="store_true", help="Include first layer or not."
    )
    parser.add_argument(
        "--max-iter", default=100000, type=int, help="Maximum iteration"
    )
    args = parser.parse_args()

    cfg = toml.load(args.cfg)
    board_cfg = toml.load(args.board)
    print(board_cfg)

    solver = SimulatedAnnealing(cfg, board_cfg, args.dsp_scale, args.incl_fst)
    sol, costs = solver.run(args.max_iter)
    print(
        solver.get_ps(sol),
        solver.get_cycles(sol),
        solver.get_dsp_usage(sol),
        solver.get_cost(sol),
        max(solver.get_cycles(sol)),
        sum(solver.get_cycles(sol)),
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(costs)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cost")
    ax.set_yscale("log")
    fig.savefig("Simulated_Annealing.png")

    ps = solver.get_ps(sol)
    for i, layer_cfg in enumerate(cfg["layers"].values()):
        if i > 0:
            layer_cfg["P_C"] = [ps[i]]
        layer_cfg["P_F"] = [ps[i + 1]]

    suffix = f"_sa_dsp_{args.dsp_scale:.2f}".replace(".", "_")
    if args.incl_fst:
        suffix += "_incl_fst"
    cfg["NAME"] += suffix

    out_file_name = args.cfg.split(".")[0] + suffix + ".toml"

    toml.dump(cfg, open(out_file_name, "w"))


if __name__ == "__main__":
    main()
