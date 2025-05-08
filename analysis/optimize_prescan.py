import os
import csv
import argparse
import numpy as np

from ray_field import CheckpointName
from ray_field.algorithm import Algorithm
from analysis import ALGORITHM_LIST, OBJECT_NAMES, model_name_dict, class_dict, model_checkpoint_dict

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/optimize_prescan/'

def compute_values(model_name: CheckpointName, algorithm: type[Algorithm]) -> tuple[list[int], list[list[int]], list[list[float]]]:
    N_values = list(range(50, 601, 50))
    M_values = list(range(8, 41, 8))
    
    distances, R_values = algorithm.optimize(model_name, N_values, M_values)

    return M_values, R_values, distances

def save_results(M_values: list[int], R_values: list[list[int]], distances: list[list[float]], model_name: str, algorithm: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    
    with open(f'{DIR_PATH}{model_name}_{algorithm}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(M_values)
        writer.writerow([len(M_values)])

        for entry in R_values:
            writer.writerow(entry)

        for entry in distances:
            writer.writerow(entry)

def load_results(model_name: str, algorithm: str) -> tuple[list[int], list[list[int]], list[list[float]]]:
    with open(f'{DIR_PATH}{model_name}_{algorithm}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        M_values = [int(value) for value in rows[0]]
        length = int(rows[1][0])
        R_values = [[int(value) for value in row] for row in rows[2:2+length]]
        distances = [[float(value) for value in row] for row in rows[2+length:]]
    
    return M_values, R_values, distances

def plot_results(M_values: list[int], R_values: list[list[int]], distances: list[list[float]], model_name: str, algorithm: str) -> None:
    fig, ax = plt.subplots()
    distances = np.array(distances) * 100

    for m, r, dist in zip(M_values, R_values, distances):
        ax.plot(r, dist, label=f'{m}')
    
    ax.set_ylabel('CD$\\cdot10^2$')
    #ax.set_xlim([0, 150000])
    #ax.set_ylim([1.3, 1.6])
    ax.set_xlabel('|R|')
    ax.set_title(f'{model_name_dict[model_name]}')
    ax.legend(loc=(1.04, 0), title='Prescan $N$')
    plt.grid(linestyle='dotted', color='grey')
    fig.savefig(f'{DIR_PATH}{model_name}_{algorithm}.png', bbox_inches="tight")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')
parser.add_argument("-f", "--Filename", type=str)
parser.add_argument("-a", "--Algorithm", type=str)

args = parser.parse_args()

if not args.Filename:
    print('A filename must be specified')
    exit()
elif not args.Algorithm:
    print('An algorithm must be specified')
    exit()

if args.Filename not in OBJECT_NAMES:
    print(f'"{args.Filename}" is not a valid key, valid keys are: {", ".join(OBJECT_NAMES)}')
    exit()
elif args.Algorithm not in ALGORITHM_LIST:
    print(f'"{args.Algorithm}" is not a valid key, valid keys are: {", ".join(ALGORITHM_LIST)}')
    exit()

if args.Load:
    M_values, R_values, distances = load_results(args.Filename, args.Algorithm)
    plot_results(M_values, R_values, distances, args.Filename, args.Algorithm)
elif args.Save:
    M_values, R_values, distances = compute_values(model_checkpoint_dict[args.Filename], class_dict[args.Algorithm])
    save_results(M_values, R_values, distances, args.Filename, args.Algorithm)
    plot_results(M_values, R_values, distances, args.Filename, args.Algorithm)
else:
    print('Neither "-s" or "-l" specified')
