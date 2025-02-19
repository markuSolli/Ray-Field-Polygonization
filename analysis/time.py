import csv
import argparse

from ray_field.algorithm import Algorithm
from analysis import ALGORITHM_LIST, N_VALUES, OBJECT_NAMES, class_dict

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/time'

def compute_values(algorithm: type[Algorithm]) -> tuple[list[str], list[list[int]], list[list[float]]]:
    times = []
    R_values = []

    # Warm up
    algorithm.surface_reconstruction(OBJECT_NAMES[0], 100)

    for i in range(len(OBJECT_NAMES)):
        print(OBJECT_NAMES[i])
        t, R = algorithm.time(OBJECT_NAMES[i], N_VALUES)
        times.append(t)
        R_values.append(R)

    return OBJECT_NAMES, R_values, times

def save_results(object_names: list[str], R_values: list[list[int]], times: list[list[float]], algorithm: str) -> None:
    with open(f'{DIR_PATH}_{algorithm}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(object_names)
        writer.writerow([len(R_values)])
        
        for entry in R_values:
            writer.writerow(entry)

        for entry in times:
            writer.writerow(entry)

def load_results(algorithm: str) -> tuple[list[str], list[list[int]], list[list[float]]]:
    with open(f'{DIR_PATH}_{algorithm}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        object_names = rows[0]
        length = int(rows[1][0])
        R_values = [[int(value) for value in row] for row in rows[2:2+length]]
        times = [[float(value) for value in row] for row in rows[2+length:]]
    
    return object_names, R_values, times

def plot_results(object_names: list[str], R_values: list[list[int]], times: list[list[float]], algorithm: str) -> None:
    fig, ax = plt.subplots()

    for i, entry in enumerate(times):
        ax.plot(R_values[i], entry, label=f'{object_names[i]}')
    
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('|R|')
    ax.set_title(algorithm)
    ax.legend(loc=(1.04, 0), title='Object')
    plt.grid(linestyle='dotted', color='grey')
    fig.savefig(f'{DIR_PATH}_{algorithm}.png', bbox_inches="tight")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')
parser.add_argument("-a", "--Algorithm", type=str)

args = parser.parse_args()

if not args.Algorithm:
    print('An algorithm must be specified')
    exit()

if args.Algorithm not in ALGORITHM_LIST:
    print(f'"{args.Algorithm}" is not a valid key, valid keys are: {", ".join(ALGORITHM_LIST)}')
    exit()

if args.Load:
    object_names, N_values, times = load_results(args.Algorithm)
    plot_results(object_names, N_values, times, args.Algorithm)
elif args.Save:
    object_names, N_values, times = compute_values(class_dict[args.Algorithm])
    save_results(object_names, N_values, times, args.Algorithm)
    plot_results(object_names, N_values, times, args.Algorithm)