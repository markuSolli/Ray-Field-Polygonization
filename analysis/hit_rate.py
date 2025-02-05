import csv
import argparse

from ray_field import prescan_cone, baseline
from analysis import ALGORITHM_LIST, N_VALUES, OBJECT_NAMES

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/hit_rate'

def compute_values_baseline() -> tuple[list[str], list[int], list[list[float]]]:
    hit_rates = []

    for i in range(len(OBJECT_NAMES)):
        print(OBJECT_NAMES[i])
        result = baseline.baseline_hit_rate(OBJECT_NAMES[i], N_VALUES)
        hit_rates.append(result)

    return OBJECT_NAMES, N_VALUES, hit_rates

def compute_values_prescan_cone() -> tuple[list[str], list[int], list[list[float]]]:
    hit_rates = []

    for i in range(len(OBJECT_NAMES)):
        print(OBJECT_NAMES[i])
        result = prescan_cone.prescan_cone_hit_rate(OBJECT_NAMES[i], N_VALUES)
        hit_rates.append(result)

    return OBJECT_NAMES, N_VALUES, hit_rates

def save_results(object_names: list[str], N_values: list[int], hit_rates: list[list[float]], algorithm: str) -> None:
    with open(f'{DIR_PATH}_{algorithm}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(object_names)
        writer.writerow(N_values)

        for entry in hit_rates:
            writer.writerow(entry)

def load_results(algorithm: str) -> tuple[list[str], list[int], list[list[float]]]:
    with open(f'{DIR_PATH}_{algorithm}.csv', mode='r') as file:
        reader = csv.reader(file)

        object_names = next(reader)
        N_values = next(reader)
        N_values = [int (x) for x in N_values]

        hit_rates = []
        i = 0
        for row in reader:
            hit_rates.append([])

            for entry in row:
                hit_rates[i].append(float(entry))
            
            i += 1
    
    return object_names, N_values, hit_rates

def plot_results(object_names: list[str], N_values: list[int], hit_rates: list[list[float]], algorithm: str) -> None:
    fig, ax = plt.subplots()

    for i, entry in enumerate(hit_rates):
        ax.plot(N_values, entry, label=f'{object_names[i]}')
    
    ax.set_ylabel('Hit Rate')
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 1.0])
    ax.set_xlabel('N')
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
    object_names, N_values, hit_rates = load_results(args.Algorithm)
    plot_results(object_names, N_values, hit_rates, args.Algorithm)
elif args.Save:
    if args.Algorithm == 'baseline':
        object_names, N_values, hit_rates = compute_values_baseline()
    elif args.Algorithm == 'prescan_cone':
        object_names, N_values, hit_rates = compute_values_prescan_cone()
    else:
        exit()
    
    save_results(object_names, N_values, hit_rates, args.Algorithm)
    plot_results(object_names, N_values, hit_rates, args.Algorithm)