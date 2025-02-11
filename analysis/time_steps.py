import os
import csv
import argparse
import numpy as np

from ray_field import CheckpointName
from ray_field.algorithm import Algorithm
from analysis import ALGORITHM_LIST, N_VALUES, OBJECT_NAMES, class_dict

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/time_steps/'

def compute_values(model_name: CheckpointName, algorithm: type[Algorithm]) -> tuple[list[str], list[int], list[list[float]]]:
    stages = ['Ray generation', 'MARF query', 'Surface reconstruction']
    
    # Warmup
    algorithm.surface_reconstruction(model_name, 100)

    times = algorithm.time_steps(model_name, N_VALUES)

    return stages, N_VALUES, times

def save_results(stages: list[str], N_values: list[int], times: list[list[float]], algorithm: str, model_name: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    
    with open(f'{DIR_PATH}{algorithm}_{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(stages)
        writer.writerow(N_values)

        for entry in times:
            writer.writerow(entry)

def load_results(algorithm: str, model_name: str) -> tuple[list[int], list[int], list[list[float]]]:
    with open(f'{DIR_PATH}{algorithm}_{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        stages = rows[0]
        N_values = [int(value) for value in rows[1]]
        times = [[float(value) for value in row] for row in rows[2:]]
    
    return stages, N_values, times

def plot_results(stages: list[str], N_values: list[int], times: list[list[float]], algorithm: str, model_name: str) -> None:
    times = np.array(times).T

    fig, ax = plt.subplots()

    ax.stackplot(N_values, times, labels=stages)
    
    ax.set_ylabel('Time (s)')
    ax.set_xlim([0, N_values[-1]])
    #ax.set_ylim([0, 6.5])
    ax.set_xlabel('N')
    ax.set_title(f'{algorithm} - {model_name}')
    ax.legend(loc=(1.04, 0), title='Object')
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