import os
import csv
import argparse
import numpy as np

from ray_field import CheckpointName
from ray_field.angle_filter import AngleFilter
from analysis import N_VALUES, OBJECT_NAMES, model_name_dict, model_checkpoint_dict

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/filter_dist/'

def compute_values(model_name: CheckpointName) -> tuple[list[float], list[int], list[list[float]]]:
    M_values = [0.0, -0.2, -0.4, -0.6, -0.8]
    
    distances, R_values = AngleFilter.optimize(model_name, N_VALUES, M_values)

    return M_values, R_values, distances

def save_results(M_values: list[float], R_values: list[int], distances: list[list[float]], model_name: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    
    with open(f'{DIR_PATH}{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(M_values)
        writer.writerow(R_values)

        for entry in distances:
            writer.writerow(entry)

def load_results(model_name: str) -> tuple[list[int], list[int], list[list[float]]]:
    with open(f'{DIR_PATH}{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        M_values = [float(value) for value in rows[0]]
        R_values = [int(value) for value in rows[1]]
        distances = [[float(value) for value in row] for row in rows[2:]]
    
    return M_values, R_values, distances

def plot_results(M_values: list[float], R_values: list[int], distances: list[list[float]], model_name: str) -> None:
    fig, ax = plt.subplots()
    distances = np.array(distances) * 100

    for i, entry in enumerate(distances):
        ax.plot(R_values, entry, label=f'{M_values[i]:.1f}')
    
    ax.set_ylabel('CD$\\cdot10^2$')
    #ax.set_xlim([0, 150000])
    ax.set_ylim([1.2, 2.0])
    ax.set_xlabel('|R|')
    ax.set_title(f'{model_name_dict[model_name]}')
    ax.legend(loc=(1.04, 0), title='Cosine limit')
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
    M_values, R_values, distances = load_results(args.Filename)
    plot_results(M_values, R_values, distances, args.Filename)
elif args.Save:
    M_values, R_values, distances = compute_values(model_checkpoint_dict[args.Filename])
    save_results(M_values, R_values, distances, args.Filename)
    plot_results(M_values, R_values, distances, args.Filename)
else:
    print('Neither "-s" or "-l" specified')
