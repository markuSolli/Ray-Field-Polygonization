import csv
import torch
import gc
import argparse

from ray_field import BUNNY, BUDDHA, ARMADILLO, DRAGON, LUCY
from ray_field import utils
from ifield.models import intersection_fields

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

DIR_PATH = 'analysis/data/surface_area'

model_dict = {
    'Bunny': BUNNY,
    'Buddha': BUDDHA,
    'Armadillo': ARMADILLO,
    'Dragon': DRAGON,
    'Lucy': LUCY
}

def compute_values() -> tuple[list[str], list[int], list[float]]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    object_names = ['Bunny', 'Buddha', 'Armadillo', 'Dragon', 'Lucy']
    N_values = list(range(100, 1001, 100))
    surface_areas = []

    for _ in object_names:
        surface_areas.append([])

    for i in range(len(object_names)):
        print(object_names[i])
        model = intersection_fields.IntersectionFieldAutoDecoderModel.load_from_checkpoint(model_dict[object_names[i]])
        model.eval().to(device)

        for N in N_values:
            print(N, end=':\t')

            origins, dirs = utils.generate_rays_between_sphere_points(N)
            origins = origins.to(device)
            dirs = dirs.to(device)

            result = model.forward(dict(origins=origins, dirs=dirs), intersections_only = False)

            intersections = result[2].cpu()
            intersection_normals = result[3].cpu()
            is_intersecting = result[4].cpu()

            is_intersecting = torch.flatten(is_intersecting)
            intersections = torch.flatten(intersections, end_dim=2)[is_intersecting].detach().numpy()
            intersection_normals = torch.flatten(intersection_normals, end_dim=2)[is_intersecting].detach().numpy()

            surface_area = utils.poisson_surface_reconstruction(intersections, intersection_normals, 8).get_surface_area()
            surface_areas[i].append(surface_area)

            print(surface_area)

            del origins, dirs, result
            torch.cuda.empty_cache()
            gc.collect()
        
        model.cpu()
        del model
        torch.cuda.empty_cache()

    return object_names, N_values, surface_areas

def save_results(object_names: list[str], N_values: list[int], surface_areas: list[list[float]]) -> None:
    with open(f'{DIR_PATH}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(object_names)
        writer.writerow(N_values)

        for entry in surface_areas:
            writer.writerow(entry)

def load_results() -> tuple[list[str], list[int], list[list[float]]]:
    with open(f'{DIR_PATH}.csv', mode='r') as file:
        reader = csv.reader(file)

        object_names = next(reader)
        N_values = next(reader)
        N_values = [int (x) for x in N_values]

        surface_areas = []
        i = 0
        for row in reader:
            surface_areas.append([])

            for entry in row:
                surface_areas[i].append(float(entry))
            
            i += 1
    
    return object_names, N_values, surface_areas

def plot_results(object_names: list[str], N_values: list[int], surface_areas: list[list[float]]) -> None:
    fig, ax = plt.subplots()

    for i, entry in enumerate(surface_areas):
        ax.plot(N_values, entry, label=f'{object_names[i]}')
    
    ax.set_ylabel('Surface Area')
    ax.set_xlabel('N')
    ax.legend(loc=(1.04, 0), title='Object')
    plt.grid(linestyle='dotted', color='grey')
    fig.savefig(f'{DIR_PATH}.png', bbox_inches="tight")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')

args = parser.parse_args()

if args.Load:
    object_names, N_values, surface_areas = load_results()
    plot_results(object_names, N_values, surface_areas)
elif args.Save:
    object_names, N_values, surface_areas = compute_values()
    save_results(object_names, N_values, surface_areas)
    plot_results(object_names, N_values, surface_areas)