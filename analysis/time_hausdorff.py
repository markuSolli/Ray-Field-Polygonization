import os
import csv
import glob
import argparse

from ray_field import CheckpointName
from ray_field.baseline_device import BaselineDevice
from ray_field.algorithm import Algorithm
from analysis import ALGORITHM_LIST, N_VALUES, OBJECT_NAMES, class_dict

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/time_hausdorff/'

def compute_values(algorithm: type[Algorithm], model_name: CheckpointName) -> tuple[list[float], list[float], list[int]]:
    # Warm up
    BaselineDevice.surface_reconstruction(model_name, 100)

    return algorithm.time_hausdorff(model_name, N_VALUES)

def save_results(times: list[float], distances: list[float], R_values: list[int], algorithm: str, model_name: str) -> None:
    with open(f'{DIR_PATH}{algorithm}_{model_name}.csv', mode='a', newline='') as file:
        writer = csv.writer(file)

        for r, t, d in zip(R_values, times, distances):
            writer.writerow([r, t, d])

def create_results(times: list[float], distances: list[float], R_values: list[int], algorithm: str, model_name: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    
    with open(f'{DIR_PATH}{algorithm}_{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        for r, t, d in zip(R_values, times, distances):
            writer.writerow([r, t, d])

def load_results(model_name: str) -> tuple[list[str], list[list[float]], list[list[float]], list[list[int]]]:
    algorithms = []
    all_times = []
    all_distances = []
    all_R_values = []

    pattern = f"{DIR_PATH}*_{model_name}.csv"
    file_paths = glob.glob(pattern)

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        algorithm = filename[: -len((f'_{model_name}.csv'))]
        algorithms.append(algorithm)

        times = []
        distances = []
        R_values = []

        with open(file_path, mode='r') as file:
            reader = csv.reader(file)

            for row in reader:
                R_values.append(int(row[0]))
                times.append(float(row[1]))
                distances.append(float(row[2]))

        all_times.append(times)
        all_distances.append(distances)
        all_R_values.append(R_values)

    return algorithms, all_times, all_distances, all_R_values

def plot_results(algorithms: list[str], times: list[list[float]], distances: list[list[float]], R_values: list[list[int]], model_name: str) -> None:
    fig, ax = plt.subplots()

    for i, entry in enumerate(times):
        ax.scatter(entry, distances[i], label=f'{algorithms[i]}', alpha=0.5)
    
    ax.set_ylabel('Hausdorff Distance')
    ax.set_xlabel('Time (s)')
    ax.set_ylim(0)
    ax.set_title(model_name)
    ax.legend(loc=(1.04, 0), title='Object')
    plt.grid(linestyle='dotted', color='grey')
    fig.savefig(f'{DIR_PATH}{model_name}.png', bbox_inches="tight")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')
parser.add_argument("-c", "--Create", action='store_true')
parser.add_argument("-a", "--Algorithm", type=str)
parser.add_argument("-f", "--Filename", type=str)

args = parser.parse_args()


if not args.Filename:
    print('A filename must be specified')
    exit()
elif args.Filename not in OBJECT_NAMES:
    print(f'"{args.Filename}" is not a valid key, valid keys are: {", ".join(OBJECT_NAMES)}')
    exit()

if args.Load:
    algorithms, times, distances, R_values = load_results(args.Filename)
    plot_results(algorithms, times, distances, R_values, args.Filename)
    exit()

if not args.Algorithm:
    print('An algorithm must be specified')
    exit()
if args.Algorithm not in ALGORITHM_LIST:
    print(f'"{args.Algorithm}" is not a valid key, valid keys are: {", ".join(ALGORITHM_LIST)}')
    exit()

if args.Save:
    times, distances, R_values = compute_values(class_dict[args.Algorithm], args.Filename)
    save_results(times, distances, R_values, args.Algorithm, args.Filename)
elif args.Create:
    times, distances, R_values = compute_values(class_dict[args.Algorithm], args.Filename)
    create_results(times, distances, R_values, args.Algorithm, args.Filename)
