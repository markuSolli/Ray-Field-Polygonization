import os
import csv
import argparse

from ray_field.prescan_cone import PrescanCone
from analysis import N_VALUES, OBJECT_NAMES

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/optimize/'

def compute_values(model_name: str) -> tuple[list[int], list[int], list[list[float]]]:
    M_values = list(range(15, 76, 15))
    distances = PrescanCone.optimize(model_name, N_VALUES, M_values)

    return M_values, N_VALUES, distances

def save_results(M_values: list[int], N_values: list[int], distances: list[list[float]], model_name: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    
    with open(f'{DIR_PATH}{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(M_values)
        writer.writerow(N_values)

        for entry in distances:
            writer.writerow(entry)

def load_results(model_name: str) -> tuple[list[int], list[int], list[list[float]]]:
    with open(f'{DIR_PATH}{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        M_values = [int(value) for value in rows[0]]
        N_values = [int(value) for value in rows[1]]
        distances = [[float(value) for value in row] for row in rows[2:]]
    
    return M_values, N_values, distances

def plot_results(M_values: list[int], N_values: list[int], distances: list[list[float]], model_name: str) -> None:
    fig, ax = plt.subplots()

    for i, entry in enumerate(distances):
        ax.plot(N_values, entry, label=f'{M_values[i]}')
    
    ax.set_ylabel('Chamfer Distance')
    ax.set_xlim([0, N_values[-1]])
    ax.set_xlabel('N')
    ax.set_title(model_name)
    ax.legend(loc=(1.04, 0), title='Broad scan N')
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
    M_values, N_values, distances = load_results(args.Filename)
    plot_results(M_values, N_values, distances, args.Filename)
elif args.Save:
    M_values, N_values, distances = compute_values(args.Filename)
    save_results(M_values, N_values, distances, args.Filename)
    plot_results(M_values, N_values, distances, args.Filename)