import os
import csv
import argparse
import numpy as np

from analysis import OBJECT_NAMES, model_name_dict

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/chamfer_combined/'
TIME_PATH = 'analysis/data/chamfer_time/'
DIST_PATH = 'analysis/data/chamfer_dist/'

def save_results(algorithms: list[str], times: list[list[float]], distances: list[list[float]], model_name: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    
    with open(f'{DIR_PATH}{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(algorithms)
        writer.writerow([len(algorithms)])

        for entry in times:
            writer.writerow(entry)

        for entry in distances:
            writer.writerow(entry)

def load_raw_results(model_name: str) -> tuple[list[float], list[list[float]], list[list[float]]]:
    with open(f'{TIME_PATH}{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        algorithms_time = rows[0]
        length = int(rows[1][0])
        R_values_time = [[int(value) for value in row] for row in rows[2:2+length]]
        times = [[float(value) for value in row] for row in rows[2+length:]]
    
    with open(f'{DIST_PATH}{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        algorithms_dist = rows[0]
        length = int(rows[1][0])
        R_values_dist = [[int(value) for value in row] for row in rows[2:2+length]]
        distances = [[float(value) for value in row] for row in rows[2+length:]]
    
    assert R_values_time == R_values_dist, 'R_values mismatch'
    assert algorithms_time == algorithms_dist, 'Algorithms mismatch'
    
    return algorithms_time, times, distances

def load_results(model_name: str) -> tuple[list[float], list[list[float]], list[list[float]]]:
    with open(f'{DIR_PATH}{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        algorithms = rows[0]
        length = int(rows[1][0])
        times = [[float(value) for value in row] for row in rows[2:2+length]]
        distances = [[float(value) for value in row] for row in rows[2+length:]]
    
    return algorithms, times, distances

def plot_results(algorithms: list[str], times: list[list[float]], distances: list[list[float]], model_name: str) -> None:
    fig, ax = plt.subplots()

    distances = np.array(distances) * 100

    for i in range(len(algorithms)):
        ax.plot(times[i], distances[i], label=algorithms[i])
    
    ax.set_ylabel('CD$\\cdot10^2$')
    #ax.set_xlim([0, 3.5])
    #ax.set_ylim([1.5, 3.5])
    ax.set_xlabel('Time (s)')
    ax.set_title(f'{model_name_dict[model_name]}')
    ax.legend(loc=(1.04, 0), title='Algorithm')
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
    algorithms, times, distances = load_results(args.Filename)
    plot_results(algorithms, times, distances, args.Filename)
elif args.Save:
    algorithms, times, distances = load_raw_results(args.Filename)
    save_results(algorithms, times, distances, args.Filename)
    plot_results(algorithms, times, distances, args.Filename)
else:
    print('Neither "-s" or "-l" specified')
