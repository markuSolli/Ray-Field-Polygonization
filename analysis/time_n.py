import os
import csv
import torch
import gc
import argparse
import numpy as np

from ray_field import BUNNY, BUDDHA, ARMADILLO, DRAGON, LUCY
from ray_field import utils
from ifield.models import intersection_fields
from timeit import default_timer as timer

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/time_n/'

model_dict = {
    'Bunny': BUNNY,
    'Buddha': BUDDHA,
    'Armadillo': ARMADILLO,
    'Dragon': DRAGON,
    'Lucy': LUCY
}

def compute_values(filename: str) -> tuple[list[int], list[list[float]]]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = intersection_fields.IntersectionFieldAutoDecoderModel.load_from_checkpoint(model_dict[filename])
    model.eval().to(device)
    
    N_values = list(range(100, 1001, 100))
    times: list[list[float]] = []

    # Warmup
    torch.zeros(1, device='cuda')
    torch.cuda.empty_cache()
    gc.collect()
    
    for i in range(len(N_values)):
        print(f'N: {N_values[i]}')
        n_times = [0.0, 0.0, 0.0]

        for j in range(5):
            ray_start = timer()

            origins, dirs = utils.generate_rays_between_sphere_points(N_values[i])
            origins = origins.to(device)
            dirs = dirs.to(device)

            ray_end = timer()

            with torch.no_grad():
                result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)

                intersections = result[2].cpu()
                intersection_normals = result[3].cpu()
                is_intersecting = result[4].cpu()

                is_intersecting = torch.flatten(is_intersecting)
                intersections = torch.flatten(intersections, end_dim=2)[is_intersecting].detach().numpy()
                intersection_normals = torch.flatten(intersection_normals, end_dim=2)[is_intersecting].detach().numpy()

            query_end = timer()

            utils.poisson_surface_reconstruction(intersections, intersection_normals, 8)

            poisson_end = timer()

            n_times[0] = n_times[0] + (ray_end - ray_start)
            n_times[1] = n_times[1] + (query_end - ray_end)
            n_times[2] = n_times[2] + (poisson_end - query_end)

            print(f'{(ray_end - ray_start):.3f}\t{(query_end - ray_end):.3f}\t{(poisson_end - query_end):.3f}')

            del origins, dirs, result, intersections, intersection_normals, is_intersecting
            torch.cuda.empty_cache()
            gc.collect()
    
        n_times = [n_time / 5 for n_time in n_times]
        times.append(n_times)

    return N_values, times

def save_results(N_values: list[int], times: list[list[float]], filename: str) -> None:
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)

    with open(f'{DIR_PATH}{filename}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(N_values)

        for entry in times:
            writer.writerow(entry)

def load_results(filename: str) -> tuple[list[int], list[list[float]]]:
    with open(f'{DIR_PATH}{filename}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        N_values = [int(value) for value in rows[0]]
        times = [[float(value) for value in row] for row in rows[1:]]
    
    return N_values, times

def plot_results(N_values: list[int], times: list[list[float]],  filename: str) -> None:
    times = np.array(times).T

    fig, ax = plt.subplots()

    ax.stackplot(N_values, times, labels=['Ray Generation', 'MARF Query', 'Surface Reconstruction'])
    
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('N')
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 6])
    ax.set_title(filename)
    ax.legend(loc=(1.04, 0))
    plt.grid(linestyle='dotted', color='grey')
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
    N_values, times = load_results(args.Filename)
    plot_results(N_values, times, args.Filename)
elif args.Save:
    N_values, times = compute_values(args.Filename)
    save_results(N_values, times, args.Filename)
    plot_results(N_values, times, args.Filename)