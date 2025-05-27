import os
import csv
import argparse
import numpy as np

from analysis import OBJECT_NAMES, model_name_dict

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/old_combined/'
TIME_PATH = 'analysis/data/chamfer_time/'
OLD_TIME_PATH = 'analysis/data/old_time/'
DIST_PATH = 'analysis/data/chamfer_dist/'

def save_results(algorithms: list[str], times: list[list[float]], distances: list[float], model_name: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
    
    with open(f'{DIR_PATH}{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(algorithms)
        writer.writerow(distances)

        for entry in times:
            writer.writerow(entry)

def load_raw_results(model_name: str) -> tuple[list[float], list[list[float]], list[float]]:
    with open(f'{OLD_TIME_PATH}{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        algorithms_old_time = rows[0]
        length = int(rows[1][0])
        R_values_old_time = [[int(value) for value in row] for row in rows[2:2+length]]
        old_times = [[float(value) for value in row] for row in rows[2+length:]]
    
    with open(f'{TIME_PATH}{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        algorithms_time = rows[0]
        length = int(rows[1][0])
        R_values_time = [[int(value) for value in row] for row in rows[2:2+length]]
        times = [[float(value) for value in row] for row in rows[2+length:]]

    time_index = algorithms_time.index('Baseline')
    new_times = times[time_index]
    old_times.append(new_times)
    algorithms_old_time.append('Parallel')
    
    with open(f'{DIST_PATH}{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        algorithms_dist = rows[0]
        length = int(rows[1][0])
        R_values_dist = [[int(value) for value in row] for row in rows[2:2+length]]
        distances = [[float(value) for value in row] for row in rows[2+length:]]
    
    dist_index = algorithms_dist.index('Baseline')
    new_distances = distances[dist_index]
    
    assert (R_values_old_time[0] == R_values_dist[dist_index] and R_values_old_time[0] == R_values_time[time_index]), 'R_values mismatch'
    
    return algorithms_old_time, old_times, new_distances

def load_results(model_name: str) -> tuple[list[float], list[list[float]], list[float]]:
    with open(f'{DIR_PATH}{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        algorithms = rows[0]
        distances = [float(value) for value in rows[1]]
        times = [[float(value) for value in row] for row in rows[2:]]
    
    return algorithms, times, distances

def plot_results(algorithms: list[str], times: list[list[float]], distances: list[float], model_name: str) -> None:
    fig, ax = plt.subplots()

    distances = np.array(distances) * 100

    for i in range(len(algorithms)):
        ax.plot(times[i], distances, label=algorithms[i])
    
    ax.set_ylabel('CD$\\cdot10^2$')
    ax.set_xlim([0, 5.0])
    #ax.set_ylim([1.5, 3.5])
    ax.set_xlabel('Time (s)')
    ax.set_title(f'{model_name_dict[model_name]}')
    ax.legend(loc=(1.04, 0), title='Implementation')
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
