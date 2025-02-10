import csv
import argparse

from ray_field import prescan_cone, baseline
from analysis import OBJECT_NAMES

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/radius'

def compute_values() -> tuple[list[str], int, float, list[list[float]]]:
    N = 1000
    baseline_result = baseline.baseline_radius(N)
    radii = []

    for i in range(len(OBJECT_NAMES)):
        print(OBJECT_NAMES[i], end='\t')
        result = prescan_cone.prescan_cone_radius(OBJECT_NAMES[i], N)
        radii.append(result)

    return OBJECT_NAMES, N, baseline_result, radii

def save_results(object_names: list[str], N: int, base_res: float, radii: list[list[float]]) -> None:
    with open(f'{DIR_PATH}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(object_names)
        writer.writerow([N])
        writer.writerow([base_res])

        for entry in radii:
            writer.writerow(entry)

def load_results() -> tuple[list[str], int, float, list[list[float]]]:
    with open(f'{DIR_PATH}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        
        object_names = rows[0]
        N = int(rows[1][0])
        base_res = float(rows[2][0])
        radii = [[float(value) for value in row] for row in rows[3:]]
    
    return object_names, N, base_res, radii

def plot_results(object_names: list[str], N: int, base_res: float, radii: list[list[float]]) -> None:
    fig, ax = plt.subplots()

    ax.violinplot(radii, showmedians=True)
    ax.axhline(base_res, color='tab:orange', linestyle = '-') 

    ax.set_ylabel('Radius')
    ax.set_xticks(range(1, len(object_names) + 1))
    ax.set_xticklabels(object_names)
    ax.set_title(f'N: {N}')

    patch0 = mpatches.Patch(color='tab:blue', label='prescan_cone')
    patch1 = mpatches.Patch(color='tab:orange', label='baseline')
    plt.legend(handles=[patch0, patch1], loc=(1.04, 0), title='Algorithm')
    plt.grid(linestyle='dotted', color='grey', axis='y')
    fig.savefig(f'{DIR_PATH}.png', bbox_inches="tight")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')

args = parser.parse_args()

if args.Load:
    object_names, N, base_res, hit_rates = load_results()
    plot_results(object_names, N, base_res, hit_rates)
elif args.Save:
    object_names, N, base_res, hit_rates = compute_values()
    save_results(object_names, N, base_res, hit_rates)
    plot_results(object_names, N, base_res, hit_rates)