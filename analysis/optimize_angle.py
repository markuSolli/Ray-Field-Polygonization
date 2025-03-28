import os
import csv
import argparse
import numpy as np

from ray_field import CheckpointName
from ray_field.angle_filter import AngleFilter
from analysis import N_VALUES, OBJECT_NAMES

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/optimize_angle/'

def compute_values(model_name: CheckpointName) -> tuple[list[float], list[list[int]], list[list[float]]]:
    M_values = np.arange(-0.9, 0.01, 0.1)
    
    distances, R_values = AngleFilter.optimize(model_name, N_VALUES, M_values)

    return M_values, R_values, distances

def save_results(M_values: list[float], R_values: list[list[int]], distances: list[list[float]], model_name: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    
    with open(f'{DIR_PATH}{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(M_values)
        writer.writerow([len(R_values)])
        
        for entry in R_values:
            writer.writerow(entry)

        for entry in distances:
            writer.writerow(entry)

def load_results(model_name: str) -> tuple[list[str], list[list[int]], list[list[float]]]:
    with open(f'{DIR_PATH}{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        M_values = [float(value) for value in rows[0]]
        length = int(rows[1][0])
        R_values = [[int(value) for value in row] for row in rows[2:2+length]]
        distances = [[float(value) for value in row] for row in rows[2+length:]]
    
    return M_values, R_values, distances

def plot_results(M_values: list[float], R_values: list[list[int]], distances: list[list[float]], model_name: str) -> None:
    fig, ax = plt.subplots()

    for i, entry in enumerate(distances):
        ax.plot(R_values[i], entry, label=f'{round(M_values[i], 1)}')
    
    ax.set_ylabel('Chamfer Distance')
    ax.set_ylim([0, 0.03])
    ax.set_xlim([0, 250000])
    ax.set_xlabel('|R|')
    ax.set_title(f'{model_name}')
    ax.legend(loc=(1.04, 0), title='Angle limit')
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
    M_values, R_values, distances = compute_values(args.Filename)
    save_results(M_values, R_values, distances, args.Filename)
    plot_results(M_values, R_values, distances, args.Filename)
