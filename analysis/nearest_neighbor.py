import os
import csv
import torch
import gc
import argparse
import numpy as np

from ray_field import BUNNY, BUDDHA, ARMADILLO, DRAGON, LUCY
from ray_field import utils
from ifield.models import intersection_fields

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/nearest_neighbor/'

model_dict = {
    'Bunny': BUNNY,
    'Buddha': BUDDHA,
    'Armadillo': ARMADILLO,
    'Dragon': DRAGON,
    'Lucy': LUCY
}

model_areas = {
    'Bunny': 5.3,
    'Buddha': 4.5,
    'Armadillo': 4.3,
    'Dragon': 5.8,
    'Lucy': 2.5
}

def compute_values(filename: str) -> tuple[list[int], list[list[float]]]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = intersection_fields.IntersectionFieldAutoDecoderModel.load_from_checkpoint(model_dict[filename])
    model.eval().to(device)
    
    N_values = list(range(100, 1001, 100))
    nnd: list[list[float]] = []
    
    for i in range(len(N_values)):
        print(N_values[i])
        origins, dirs = utils.generate_rays_between_sphere_points(N_values[i])
        origins = origins.to(device)
        dirs = dirs.to(device)

        result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)

        intersections = result[2].cpu()
        is_intersecting = result[4].cpu()

        is_intersecting = torch.flatten(is_intersecting)
        intersections = torch.flatten(intersections, end_dim=2)[is_intersecting].detach().numpy()

        distances = utils.nearest_neighbor_distance(intersections)
        nnd.append(distances)

        del origins, dirs, result
        torch.cuda.empty_cache()
        gc.collect()
    
    return N_values, nnd

def save_results(N_values: list[int], nnd: list[list[float]], filename: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)

    with open(f'{DIR_PATH}{filename}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(N_values)

        for entry in nnd:
            writer.writerow(entry)

def load_results(filename: str) -> tuple[list[int], list[list[float]]]:
    with open(f'{DIR_PATH}{filename}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        N_values = [int(value) for value in rows[0]]
        nnd = [[float(value) for value in row] for row in rows[1:]]
    
    return N_values, nnd

def plot_results(N_values: list[int], nnd: list[list[float]],  filename: str) -> None:
    for i in range(len(nnd)):
        median = np.median(nnd[i])
        expected_distance = np.sqrt(model_areas[filename] / N_values[i])

        for j in range(len(nnd[i])):
            nnd[i][j] = (nnd[i][j] - median) / expected_distance

    fig, ax = plt.subplots()

    ax.violinplot(nnd, showmedians=True)

    ax.set_ylabel('Normalized median-shifted NND')
    ax.set_xlabel('N')
    ax.set_ylim([-0.25, 1.75])
    ax.set_xticks(range(1, len(N_values) + 1))
    ax.set_xticklabels(N_values)
    ax.set_title(filename)
    plt.grid(linestyle='dotted', color='grey', axis='y')
    fig.savefig(f'{DIR_PATH}{filename}.png', bbox_inches="tight")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')
parser.add_argument("-f", "--Filename", type=str)

args = parser.parse_args()

if not args.Filename:
    print('A filename must be specified')
    exit()

if args.Filename not in model_dict:
    print(f'"{args.Filename}" is not a valid key, valid keys are: {", ".join(model_dict.keys())}')
    exit()

if args.Load:
    N_values, nnd = load_results(args.Filename)
    plot_results(N_values, nnd, args.Filename)
elif args.Save:
    N_values, nnd = compute_values(args.Filename)
    save_results(N_values, nnd, args.Filename)
    plot_results(N_values, nnd, args.Filename)