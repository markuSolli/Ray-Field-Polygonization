import os
import csv
import argparse

from ray_field import CheckpointName
from ray_field.algorithm import Algorithm
from ray_field.baseline import Baseline
from analysis import ALGORITHM_LIST, N_VALUES, OBJECT_NAMES, class_dict

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/optimize/'

def compute_values(model_name: CheckpointName, algorithm: type[Algorithm]) -> tuple[list[int], list[int], list[list[float]]]:
    if issubclass(algorithm, Baseline):
        M_values = list(range(6, 15, 2))
    else:
        M_values = list(range(30, 151, 30))
    
    distances = algorithm.optimize(model_name, N_VALUES, M_values)

    return M_values, N_VALUES, distances

def save_results(M_values: list[int], N_values: list[int], distances: list[list[float]], algorithm: str, model_name: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    
    with open(f'{DIR_PATH}{algorithm}_{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(M_values)
        writer.writerow(N_values)

        for entry in distances:
            writer.writerow(entry)

def load_results(algorithm: str, model_name: str) -> tuple[list[int], list[int], list[list[float]]]:
    with open(f'{DIR_PATH}{algorithm}_{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        M_values = [int(value) for value in rows[0]]
        N_values = [int(value) for value in rows[1]]
        distances = [[float(value) for value in row] for row in rows[2:]]
    
    return M_values, N_values, distances

def plot_results(M_values: list[int], N_values: list[int], distances: list[list[float]], algorithm: str, model_name: str) -> None:
    fig, ax = plt.subplots()

    for i, entry in enumerate(distances):
        ax.plot(N_values, entry, label=f'{M_values[i]}')
    
    ax.set_ylabel('Chamfer Distance')
    ax.set_xlim([0, N_values[-1]])
    ax.set_ylim(0)
    ax.set_xlabel('N')
    ax.set_title(f'{algorithm} - {model_name}')
    ax.legend(loc=(1.04, 0), title='Broad scan N')
    plt.grid(linestyle='dotted', color='grey')
    fig.savefig(f'{DIR_PATH}{algorithm}_{model_name}.png', bbox_inches="tight")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')
parser.add_argument("-a", "--Algorithm", type=str)
parser.add_argument("-f", "--Filename", type=str)

args = parser.parse_args()

if not args.Algorithm:
    print('An algorithm must be specified')
    exit()
elif not args.Filename:
    print('A filename must be specified')
    exit()

if args.Algorithm not in ALGORITHM_LIST:
    print(f'"{args.Algorithm}" is not a valid key, valid keys are: {", ".join(ALGORITHM_LIST)}')
    exit()
elif args.Filename not in OBJECT_NAMES:
    print(f'"{args.Filename}" is not a valid key, valid keys are: {", ".join(OBJECT_NAMES)}')
    exit()

if args.Load:
    stages, N_values, times = load_results(args.Algorithm, args.Filename)
    plot_results(stages, N_values, times, args.Algorithm, args.Filename)
elif args.Save:
    stages, N_values, times = compute_values(args.Filename, class_dict[args.Algorithm])
    save_results(stages, N_values, times, args.Algorithm, args.Filename)
    plot_results(stages, N_values, times, args.Algorithm, args.Filename)
