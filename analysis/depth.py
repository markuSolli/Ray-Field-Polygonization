import os
import csv
import argparse

from ray_field import CheckpointName
from ray_field.baseline_device import BaselineDevice
from old_analysis import N_VALUES, OBJECT_NAMES

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/depth/'

def compute_values(model_name: CheckpointName) -> tuple[list[int], list[int], list[list[float]]]:
    D_values = list(range(6, 11, 1))
    
    distances, R_values = BaselineDevice.optimize(model_name, N_VALUES, D_values)

    return D_values, R_values, distances

def save_results(D_values: list[int], R_values: list[int], distances: list[list[float]], model_name: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    
    with open(f'{DIR_PATH}{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(D_values)
        writer.writerow(R_values)

        for entry in distances:
            writer.writerow(entry)

def load_results(model_name: str) -> tuple[list[int], list[int], list[list[float]]]:
    with open(f'{DIR_PATH}{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        D_values = [int(value) for value in rows[0]]
        R_values = [int(value) for value in rows[1]]
        distances = [[float(value) for value in row] for row in rows[2:]]
    
    return D_values, R_values, distances

def plot_results(D_values: list[int], R_values: list[int], distances: list[list[float]], model_name: str) -> None:
    fig, ax = plt.subplots()

    for i, entry in enumerate(distances):
        ax.plot(R_values, entry, label=f'{D_values[i]}')
    
    ax.set_ylabel('Chamfer Distance')
    ax.set_xlim([0, 150000])
    ax.set_ylim([0.012, 0.036])
    ax.set_xlabel('|R|')
    ax.set_title(f'{model_name.capitalize()}')
    ax.legend(loc=(1.04, 0), title='Depth')
    plt.grid(linestyle='dotted', color='grey')
    fig.savefig(f'{DIR_PATH}{model_name}.png', bbox_inches="tight")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')
parser.add_argument("-f", "--Filename", type=str)

args = parser.parse_args()

if not args.Filename:
    print('A filename must be specified')
    exit()

if args.Filename not in OBJECT_NAMES:
    print(f'"{args.Filename}" is not a valid key, valid keys are: {", ".join(OBJECT_NAMES)}')
    exit()

if args.Load:
    D_values, R_values, times = load_results(args.Filename)
    plot_results(D_values, R_values, times, args.Filename)
elif args.Save:
    D_values, R_values, times = compute_values(args.Filename)
    save_results(D_values, R_values, times, args.Filename)
    plot_results(D_values, R_values, times, args.Filename)
else:
    print('Neither "-s" or "-l" specified')
