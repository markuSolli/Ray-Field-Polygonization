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

DIR_PATH = 'analysis/data/hit_rate'

model_dict = {
    'Bunny': BUNNY,
    'Buddha': BUDDHA,
    'Armadillo': ARMADILLO,
    'Dragon': DRAGON,
    'Lucy': LUCY
}

def compute_values() -> tuple[list[str], list[int], list[list[float]]]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    object_names = ['Bunny', 'Buddha', 'Armadillo', 'Dragon', 'Lucy']
    N_values = list(range(100, 1001, 100))
    hit_rates = []

    for _ in object_names:
        hit_rates.append([])

    for i in range(len(object_names)):
        print(object_names[i])
        model = intersection_fields.IntersectionFieldAutoDecoderModel.load_from_checkpoint(model_dict[object_names[i]])
        model.eval().to(device)

        for N in N_values:
            print(str(N), end=':\t')
            sphere_points = utils.generate_equidistant_sphere_points(N)
            origins, dirs = utils.generate_rays_between_points(sphere_points)

            sphere_n = np.size(sphere_points, 0)
            rays_n = sphere_n * (sphere_n - 1)

            origins = origins.to(device)
            dirs = dirs.to(device)

            result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)

            is_intersecting = result[4].cpu().sum().item()
            hit_rate = is_intersecting / rays_n
            hit_rates[i].append(hit_rate)
            print(str(hit_rate))

            del origins, dirs, result
            torch.cuda.empty_cache()
            gc.collect()
        
        model.cpu()
        del model
        torch.cuda.empty_cache()

    return object_names, N_values, hit_rates

def save_results(object_names: list[str], N_values: list[int], hit_rates: list[list[float]]) -> None:
    with open(f'{DIR_PATH}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(object_names)
        writer.writerow(N_values)

        for entry in hit_rates:
            writer.writerow(entry)

def load_results() -> tuple[list[str], list[int], list[list[float]]]:
    with open(f'{DIR_PATH}.csv', mode='r') as file:
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

def plot_results(object_names: list[str], N_values: list[int], hit_rates: list[list[float]]) -> None:
    fig, ax = plt.subplots()

    for i, entry in enumerate(hit_rates):
        ax.plot(N_values, entry, label=f'{object_names[i]}')
    
    ax.set_ylabel('Hit Rate')
    ax.set_ylim([0, 1.0])
    ax.set_xlabel('N')
    ax.legend(loc=(1.04, 0), title='Object')
    plt.grid(linestyle='dotted', color='grey')
    fig.savefig(f'{DIR_PATH}.png', bbox_inches="tight")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')

args = parser.parse_args()

if args.Load:
    object_names, N_values, hit_rates = load_results()
    plot_results(object_names, N_values, hit_rates)
elif args.Save:
    object_names, N_values, hit_rates = compute_values()
    save_results(object_names, N_values, hit_rates)
    plot_results(object_names, N_values, hit_rates)