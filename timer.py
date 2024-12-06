import ray_field

import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from trimesh import Trimesh

POISSON_DEPTH = 8
DATA_DIR = 'timings'
OBJECT_NAME = 'spot'

def measure_time() -> tuple[list[int], list[list[float]]]:
    original_mesh: Trimesh = ray_field.get_scaled_mesh(f'models/{OBJECT_NAME}.obj')

    n_points: list[int] = list(range(100, 2001, 100))
    times: list[list[float]] = []

    for n in n_points:
        print(f'N: {n}')
        
        rays_start = timer()
        rays = ray_field.generate_rays_between_sphere_points(n)
        rays_end = timer()

        intersection_start = timer()
        intersect_locations, intersect_normals = ray_field.ray_intersection_with_mesh_batched(rays, original_mesh)
        intersection_end = timer()

        poisson_start = timer()
        generated_mesh = ray_field.poisson_surface_reconstruction(intersect_locations, intersect_normals, POISSON_DEPTH)
        poisson_end = timer()

        rays_time = rays_end - rays_start
        intersection_time = intersection_end - intersection_start
        poisson_time = poisson_end - poisson_start

        times.append([rays_time, intersection_time, poisson_time])

        print(f'Generate rays:\t\t{rays_time:.6f} s')
        print(f'Ray intersection:\t{intersection_time:.6f} s')
        print(f'Poisson:\t\t{poisson_time:.6f} s')
        print('=======================================')
    
    return n_points, times
    
def save_results(points: list[int], timings: list[list[float]], dir: str, obj_name: str) -> None:
    with open(f'data/{dir}/{obj_name}_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(points)

        for time in timings:
            writer.writerow(time)

def load_results(dir: str, obj_name: str) -> tuple[list[int], list[list[float]]]:
    with open(f'data/{dir}/{obj_name}_data.csv', mode='r') as file:
        reader = csv.reader(file)

        n_points = next(reader)
        n_points = [int (x) for x in n_points]

        timings = []
        i = 0
        for row in reader:
            timings.append([])

            for entry in row:
                timings[i].append(float(entry))
            
            i += 1
    
    return n_points, timings

def plot_results(points: list[int], timings: list[list[float]]) -> None:
    timings = np.array(timings).T

    fig, ax = plt.subplots()

    ax.stackplot(points, timings, labels=['Ray generation', 'Ray intersection', 'Poisson'])
    
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('N Points')
    ax.set_xlim([0, 2000])
    ax.set_title(OBJECT_NAME)
    ax.legend(loc=(1.04, 0))
    plt.grid(linestyle='dotted', color='grey')
    fig.savefig(f'data/{DATA_DIR}/{OBJECT_NAME}_time_plot.png', bbox_inches="tight")
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')

args = parser.parse_args()

if args.Load:
    n_points, timings = load_results(DATA_DIR, OBJECT_NAME)
    plot_results(n_points, timings)
elif args.Save:
    n_points, timings = measure_time()
    save_results(n_points, timings, DATA_DIR, OBJECT_NAME)
    plot_results(n_points, timings)