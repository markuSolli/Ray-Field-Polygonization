import os
import csv
import argparse

from ray_field import CheckpointName
from ray_field.ball_pivoting import BallPivoting
from analysis import ALGORITHM_LIST, N_VALUES, OBJECT_NAMES, class_dict

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/optimize_radii/'

def compute_values(model_name: CheckpointName) -> tuple[list[list[float]], list[int], list[list[float]]]:
    radii = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    M_values = []

    for i in range(len(radii) - 3):
        window = radii[i:i + 4]
        M_values.append(window)
    
    distances = BallPivoting.optimize_radii(model_name, N_VALUES, M_values)

    return M_values, N_VALUES, distances

def save_results(M_values: list[list[float]], N_values: list[int], distances: list[list[float]], model_name: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    
    with open(f'{DIR_PATH}{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(N_values)
        writer.writerow([len(M_values)])

        for entry in M_values:
            writer.writerow(entry)

        for entry in distances:
            writer.writerow(entry)

def load_results(model_name: str) -> tuple[list[list[float]], list[int], list[list[float]]]:
    with open(f'{DIR_PATH}{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        N_values = rows[0]
        length = int(rows[1][0])
        M_values = [[float(value) for value in row] for row in rows[2:2+length]]
        distances = [[float(value) for value in row] for row in rows[2+length:2+length*2]]
    
    return M_values, N_values, distances

def plot_results(M_values: list[list[float]], N_values: list[int], distances: list[list[float]], model_name: str) -> None:
    fig, ax = plt.subplots()

    for i, entry in enumerate(distances):
        ax.plot(N_values, entry, label=f'{M_values[i]}')
    
    ax.set_ylabel('Chamfer Distance')
    ax.set_xlim([0, N_values[-1]])
    ax.set_ylim(0)
    ax.set_xlabel('N')
    ax.set_title(model_name)
    ax.legend(loc=(1.04, 0), title='Radii')
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
    stages, N_values, times = load_results(args.Filename)
    plot_results(stages, N_values, times, args.Filename)
elif args.Save:
    stages, N_values, times = compute_values(args.Filename)
    save_results(stages, N_values, times, args.Filename)
    plot_results(stages, N_values, times, args.Filename)
