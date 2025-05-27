import os
import csv
import argparse

from ray_field import CheckpointName
from ray_field.algorithm import Algorithm
from ray_field.baseline import Baseline
from ray_field.baseline_cpu import BaselineCPU
from analysis import ALGORITHM_LIST, OBJECT_NAMES, model_name_dict, model_checkpoint_dict, class_dict

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/old_time/'
ALG_NAMES = ['Linear CPU', 'Linear GPU']

alg_name_dict = {
    'baseline': 'Linear GPU',
    'basleline_cpu': 'Linear CPU',
}

def compute_values(model_name: CheckpointName) -> tuple[list[str], list[list[int]], list[list[float]]]:
    times = []
    R_values = []

    for i, algorithm in enumerate([BaselineCPU, Baseline]):
        print(ALG_NAMES[i])

        # Warmup
        algorithm.surface_reconstruction(model_name, 100)

        t, R = algorithm.time(model_name, 10)
        times.append(t)
        R_values.append(R)

    return ALG_NAMES, R_values, times

def compute_alg_values(model_name: CheckpointName, algorithm: type[Algorithm]) -> tuple[list[int], list[float]]:
    # Warmup
    algorithm.surface_reconstruction(model_name, 100)

    times, R_values = algorithm.time(model_name, 10)

    return R_values, times

def save_results(alg_names: list[str], R_values: list[list[int]], times: list[list[float]], model_name: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)

    with open(f'{DIR_PATH}{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(alg_names)
        writer.writerow([len(R_values)])
        
        for entry in R_values:
            writer.writerow(entry)

        for entry in times:
            writer.writerow(entry)

def save_alg_results(R_values: list[int], times: list[float], model_name: str, algorithm: str) -> None:
    with open(f'{DIR_PATH}{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        alg_names = rows[0]
        length = int(rows[1][0])
        old_R_values = [[int(value) for value in row] for row in rows[2:2+length]]
        old_times = [[float(value) for value in row] for row in rows[2+length:]]
    
    alg_name = alg_name_dict[algorithm]
    alg_index = alg_names.index(alg_name)

    old_R_values[alg_index] = R_values
    old_times[alg_index] = times

    with open(f'{DIR_PATH}{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(alg_names)
        writer.writerow([len(old_R_values)])
        
        for entry in old_R_values:
            writer.writerow(entry)

        for entry in old_times:
            writer.writerow(entry)

def load_results(model_name: str) -> tuple[list[str], list[list[int]], list[list[float]]]:
    with open(f'{DIR_PATH}{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        alg_names = rows[0]
        length = int(rows[1][0])
        R_values = [[int(value) for value in row] for row in rows[2:2+length]]
        times = [[float(value) for value in row] for row in rows[2+length:]]
    
    return alg_names, R_values, times

def plot_results(alg_names: list[str], R_values: list[list[int]], times: list[list[float]], model_name: str) -> None:
    fig, ax = plt.subplots()

    for i in range(len(alg_names)):
        ax.plot(R_values[i], times[i], label=alg_names[i])
    
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('|R|')
    #ax.set_ylim([0, 0.035])
    ax.set_title(f'{model_name_dict[model_name]}')
    ax.legend(loc=(1.04, 0), title='Object')
    plt.grid(linestyle='dotted', color='grey')
    fig.savefig(f'{DIR_PATH}{model_name}.png', bbox_inches="tight")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')
parser.add_argument("-f", "--Filename", type=str)
parser.add_argument("-a", "--Algorithm", type=str)

args = parser.parse_args()

if not args.Filename:
    print('A filename must be specified')
    exit()

if args.Filename not in OBJECT_NAMES:
    print(f'"{args.Filename}" is not a valid key, valid keys are: {", ".join(OBJECT_NAMES)}')
    exit()

if args.Algorithm and args.Algorithm not in ALGORITHM_LIST:
    print(f'"{args.Algorithm}" is not a valid key, valid keys are: {", ".join(ALGORITHM_LIST)}')
    exit()

if args.Load:
    alg_names, R_values, times = load_results(args.Filename)
    plot_results(alg_names, R_values, times, args.Filename)
elif args.Save:
    if args.Algorithm:
        R_values, times = compute_alg_values(model_checkpoint_dict[args.Filename], class_dict[args.Algorithm])
        save_alg_results(R_values, times, args.Filename, args.Algorithm)
    else:
        alg_names, R_values, times = compute_values(model_checkpoint_dict[args.Filename])
        save_results(alg_names, R_values, times, args.Filename)
        plot_results(alg_names, R_values, times, args.Filename)
else:
    print('Neither "-s" or "-l" specified')