import csv
from ray_field import utils
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/equidistant_points'

def compute_values() -> tuple[list[int], list[int]]:
    N_values = list(range(1, 1000))
    n_values = []

    for N in N_values:
        points = utils.generate_equidistant_sphere_points(N)
        n_values.append(np.size(points, 0))
    
    return N_values, n_values

def save_results(N_values: list[int], n_values: list[int]) -> None:
    with open(f'{DIR_PATH}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(N_values)
        writer.writerow(n_values)

def load_results() -> tuple[list[int], list[int]]:
    with open(f'{DIR_PATH}.csv', mode='r') as file:
        reader = csv.reader(file)
        N_points = next(reader)
        n_points = next(reader)
        N_points = [int (x) for x in N_points]
        n_points = [int (x) for x in n_points]
    
    return N_points, n_points

def plot_results(N_points: list[int], n_points: list[int]) -> None:
    fig, ax = plt.subplots()
    ax.plot(N_points, n_points)
    ax.set_xlabel('N')
    ax.set_ylabel('n')
    plt.grid(linestyle='dotted', color='grey')
    fig.savefig(f'{DIR_PATH}.png')

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')

args = parser.parse_args()

if args.Load:
    N_values, n_values = load_results()
    plot_results(N_values, n_values)
elif args.Save:
    N_values, n_values = compute_values()
    save_results(N_values, n_values)
    plot_results(N_values, n_values)
