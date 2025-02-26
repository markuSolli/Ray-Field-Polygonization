import os
import csv
import argparse
import numpy as np

from ray_field import CheckpointName
from ray_field.baseline_device import BaselineDevice
from analysis import ALGORITHM_LIST, N_VALUES, OBJECT_NAMES, class_dict

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/time_chamfer/'

def compute_values(model_name: CheckpointName) -> tuple[list[str], list[list[float]], list[list[float]], list[list[int]]]:
    times = []
    distances = []
    R_values = []

    # Warm up
    BaselineDevice.surface_reconstruction(model_name, 100)

    for i in range(len(ALGORITHM_LIST)):
        print(ALGORITHM_LIST[i])
        algorithm = class_dict[ALGORITHM_LIST[i]]
        time, distance, R = algorithm.time_chamfer(model_name, N_VALUES)
        times.append(time)
        distances.append(distance)
        R_values.append(R)

    return ALGORITHM_LIST, times, distances, R_values

def save_results(algorithms: list[str], times: list[list[float]], distances: list[list[float]], R_values: list[list[int]], model_name: CheckpointName) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    
    with open(f'{DIR_PATH}{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(algorithms)
        writer.writerow([len(distances)])
        
        for entry in times:
            writer.writerow(entry)

        for entry in distances:
            writer.writerow(entry)

        for entry in R_values:
            writer.writerow(entry)

def load_results(model_name: CheckpointName) -> tuple[list[str], list[list[float]], list[list[float]], list[list[int]]]:
    with open(f'{DIR_PATH}{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        algorithms = rows[0]
        length = int(rows[1][0])
        times = [[float(value) for value in row] for row in rows[2:2+length]]
        distances = [[float(value) for value in row] for row in rows[2+length:2+length*2]]
        R_values = [[int(value) for value in row] for row in rows[2+length*2:]]
    
    return algorithms, times, distances, R_values

def plot_results(algorithms: list[str], times: list[list[float]], distances: list[list[float]], R_values: list[list[int]], model_name: CheckpointName) -> None:
    fig, ax = plt.subplots()

    for i, entry in enumerate(times):
        ax.scatter(entry, distances[i], label=f'{algorithms[i]}')
    
    ax.set_ylabel('Chamfer Distance')
    ax.set_xlabel('Time (s)')
    ax.set_ylim(0)
    ax.set_title(model_name)
    ax.legend(loc=(1.04, 0), title='Object')
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
    algorithms, times, distances, R_values = load_results(args.Filename)
    plot_results(algorithms, times, distances, R_values, args.Filename)
elif args.Save:
    algorithms, times, distances, R_values = compute_values(args.Filename)
    save_results(algorithms, times, distances, R_values, args.Filename)
    plot_results(algorithms, times, distances, R_values, args.Filename)