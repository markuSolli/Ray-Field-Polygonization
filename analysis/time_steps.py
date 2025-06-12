import os
import csv
import argparse

from ray_field import CheckpointName
from ray_field.algorithm import Algorithm
from analysis import ALGORITHM_LIST, OBJECT_NAMES, model_name_dict, model_checkpoint_dict, class_dict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/time_steps/'

alg_name_dict = {
    'baseline_device': 'Baseline',
    'prescan_cone': 'Prescan Cone',
    'candidate_sphere': 'Candidate Sphere',
    'angle_filter': 'Angle Filter'
}

step_color_dict = {
    'Ray Generation': 'tab:blue',
    'MARF Query': 'tab:orange',
    'PSR': 'tab:green',
    'Coarse Scan': 'tab:red',
    'Filter': 'tab:purple',
}

def compute_values(model_name: CheckpointName, algorithm: type[Algorithm]) -> tuple[list[str], list[int], list[list[float]]]:
    # Warmup
    algorithm.surface_reconstruction(model_name, 100)

    steps, times, R_values = algorithm.time_steps(model_name, 10)

    return steps, R_values, times

def save_results(steps: list[str], R_values: list[int], times: list[list[float]], model_name: str, algorithm: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)

    with open(f'{DIR_PATH}{algorithm}_{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(steps)
        writer.writerow(R_values)
        
        for entry in times:
            writer.writerow(entry)

def load_results(model_name: str, algorithm: str) -> tuple[list[str], list[int], list[list[float]]]:
    with open(f'{DIR_PATH}{algorithm}_{model_name}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        steps = rows[0]
        R_values = [int(value) for value in rows[1]]
        times = [[float(value) for value in row] for row in rows[2:]]
    
    return steps, R_values, times

def plot_results(steps: list[str], R_values: list[int], times: list[list[float]], model_name: str, algorithm: str) -> None:
    fig, ax = plt.subplots()
    colors = [step_color_dict[step] for step in steps]

    ax.stackplot(R_values, times, labels=steps, colors=colors)
    
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('|R|')
    #ax.set_ylim([0, 0.035])
    ax.set_title(f'{alg_name_dict[algorithm]} - {model_name_dict[model_name]}')
    ax.legend(loc=(1.04, 0), title='Step')
    plt.grid(linestyle='dotted', color='grey', axis='y')
    fig.savefig(f'{DIR_PATH}{algorithm}_{model_name}.png', bbox_inches="tight")

def create_legend() -> None:
    legend_elements = [Patch(facecolor=color, label=step) for step, color in step_color_dict.items()]

    fig, ax = plt.subplots()
    ax.axis('off')

    ax.legend(
        handles=legend_elements,
        title='Step',
        loc='center',
        frameon=True,
        borderpad=1
    )

    fig.savefig(f'{DIR_PATH}legend.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')
parser.add_argument("-c", "--Colors", action='store_true')
parser.add_argument("-f", "--Filename", type=str)
parser.add_argument("-a", "--Algorithm", type=str)

args = parser.parse_args()

if not args.Colors:
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
    steps, R_values, times = load_results(args.Filename, args.Algorithm)
    plot_results(steps, R_values, times, args.Filename, args.Algorithm)
elif args.Save:
    steps, R_values, times = compute_values(model_checkpoint_dict[args.Filename], class_dict[args.Algorithm])
    save_results(steps, R_values, times, args.Filename, args.Algorithm)
    plot_results(steps, R_values, times, args.Filename, args.Algorithm)
elif args.Colors:
    create_legend()
else:
    print('Neither "-s" or "-l" specified')