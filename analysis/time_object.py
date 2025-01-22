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

DIR_PATH = 'analysis/data/time_object'

model_dict = {
    'Bunny': BUNNY,
    'Buddha': BUDDHA,
    'Armadillo': ARMADILLO,
    'Dragon': DRAGON,
    'Lucy': LUCY
}

def compute_values() -> tuple[list[str], list[list[float]]]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    object_names = ['Bunny', 'Buddha', 'Armadillo', 'Dragon', 'Lucy']
    times = []

    # Warmup
    torch.zeros(1, device='cuda')
    torch.cuda.empty_cache()
    gc.collect()

    for i in range(len(object_names)):
        print(object_names[i])
        model = intersection_fields.IntersectionFieldAutoDecoderModel.load_from_checkpoint(model_dict[object_names[i]])
        model.eval().to(device)

        object_times = [0.0, 0.0, 0.0]

        for j in range(5):
            ray_start = timer()

            origins, dirs = utils.generate_rays_between_sphere_points(600)
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

            object_times[0] = object_times[0] + (ray_end - ray_start)
            object_times[1] = object_times[1] + (query_end - ray_end)
            object_times[2] = object_times[2] + (poisson_end - query_end)

            print(f'{(ray_end - ray_start):.3f}\t{(query_end - ray_end):.3f}\t{(poisson_end - query_end):.3f}')

            del origins, dirs, result, intersections, intersection_normals, is_intersecting
            torch.cuda.empty_cache()
            gc.collect()

        object_times = [time / 5 for time in object_times]
        times.append(object_times)
        
        model.cpu()
        del model
        torch.cuda.empty_cache()

    return object_names, times

def save_results(object_names: list[str], times: list[list[float]]) -> None:
    with open(f'{DIR_PATH}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(object_names)

        for entry in times:
            writer.writerow(entry)

def load_results() -> tuple[list[str], list[list[float]]]:
    with open(f'{DIR_PATH}.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        object_names = rows[0]
        times = [[float(value) for value in row] for row in rows[1:]]
    
    return object_names, times

def plot_results(object_names: list[str], times: list[list[float]]) -> None:
    labels = ['Ray Generation', 'MARF Query', 'Surface Reconstruction']

    # Number of groups and bars
    n_groups = len(object_names)
    n_bars = len(times[0])

    # Indices for groups
    indices = np.arange(n_groups)

    # Bar width
    bar_width = 0.2

    # Offset for each bar
    offsets = [bar_width * i for i in range(n_bars)]

    # Plot bars
    for i in range(n_bars):
        plt.bar(indices + offsets[i], [t[i] for t in times], bar_width, label=labels[i])

    # Formatting the plot
    plt.ylabel('Time (s)')
    plt.title('N = 600')
    plt.xticks(indices + bar_width, object_names)
    plt.legend(loc=(1.04, 0))

    # Show grid and plot
    plt.grid(linestyle='dotted', color='grey', axis='y')
    plt.tight_layout()
    plt.savefig(f'{DIR_PATH}.png', bbox_inches="tight")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')

args = parser.parse_args()

if args.Load:
    object_names, times = load_results()
    plot_results(object_names, times)
elif args.Save:
    object_names, times = compute_values()
    save_results(object_names, times)
    plot_results(object_names, times)