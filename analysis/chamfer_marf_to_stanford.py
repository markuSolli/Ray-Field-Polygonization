import csv
import argparse
import trimesh

from ray_field import utils
from analysis import OBJECT_NAMES

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/chamfer_marf_to_stanford'

def compute_values() -> tuple[list[str], list[float]]:
    distances = []

    for i in range(len(OBJECT_NAMES)):
        object_name = OBJECT_NAMES[i]

        marf_intersections = utils.chamfer_distance_to_marf_1(object_name)
        samples = marf_intersections.shape[0]
        
        stanford_mesh = utils.load_and_scale_stanford_mesh(object_name)
        stanford_samples = trimesh.sample.sample_surface_even(stanford_mesh, samples)[0]

        distance = utils.chamfer_distance(stanford_samples, marf_intersections)

        distances.append(distance)

        print(f'{object_name}\t{samples}\t{distance:.6f}')

    return OBJECT_NAMES, distances

def save_results(object_names: list[str], distances: list[float]) -> None:
    with open(f'{DIR_PATH}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(object_names)
        writer.writerow(distances)

def load_results() -> tuple[list[str], list[float]]:
    with open(f'{DIR_PATH}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        object_names = rows[0]
        distances = [float(value) for value in rows[1]]
    
    return object_names, distances

def plot_results(object_names: list[str], distances: list[float]) -> None:
    fig, ax = plt.subplots()

    ax.bar(object_names, distances)
    
    ax.set_ylabel('Chamfer Distance')
    ax.set_title('MARF to Stanford')
    plt.grid(linestyle='dotted', color='grey', axis='y')
    fig.savefig(f'{DIR_PATH}.png', bbox_inches="tight")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')

args = parser.parse_args()

if args.Load:
    object_names, distances = load_results()
    plot_results(object_names, distances)
elif args.Save:
    object_names, distances = compute_values()
    save_results(object_names, distances)
    plot_results(object_names, distances)