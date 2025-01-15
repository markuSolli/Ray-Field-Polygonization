import ray_field

import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from trimesh import Trimesh

N = 600
POISSON_DEPTH = 8
DATA_DIR = 'model_timings'
OBJECT_NAME = 'model_timer'

def measure_time() -> tuple[list[int], list[list[float]]]:
    original_suzanne: Trimesh = ray_field.get_scaled_mesh(f'models/suzanne.obj')
    original_spot: Trimesh = ray_field.get_scaled_mesh(f'models/spot.obj')
    original_horse: Trimesh = ray_field.get_scaled_mesh(f'models/horse.obj')
    original_list = [original_suzanne, original_spot, original_horse]
    name_list = ['Suzanne', 'Spot', 'Horse']

    times: list[float] = [0.0, 0.0, 0.0]

    for i in range(5):
        print(f'Iteration {i + 1}')
        for j in range(3):
            print(f'Model:\t{name_list[j]}')
            time_start = timer()

            rays = ray_field.generate_rays_between_sphere_points(N)
            intersect_locations, intersect_normals = ray_field.ray_intersection_with_mesh_batched(rays, original_list[j])
            generated_mesh = ray_field.poisson_surface_reconstruction(intersect_locations, intersect_normals, POISSON_DEPTH)

            time_end = timer()
            time_total = time_end - time_start

            times[j] += time_total

            print(f'Time:\t{time_total} s')
            print('-------------------------')
        print('============================')
    
    times = [(x / 5.0) for x in times]
    
    return times
    
def save_results(timings: list[float], dir: str) -> None:
    with open(f'data/{dir}/{OBJECT_NAME}_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(timings)

def load_results(dir: str) -> list[float]:
    with open(f'data/{dir}/{OBJECT_NAME}_data.csv', mode='r') as file:
        reader = csv.reader(file)

        timings = next(reader)
    
    return timings

def plot_results(timings: list[float]) -> None:
    vertices = [507, 2930, 48485]

    fig, ax = plt.plot(vertices, timings)
    
    ax.set_ylabel('Average time (s)')
    ax.set_xlabel('Vertices')
    ax.set_xlim([0, 48485])
    ax.set_title('Execution time for vertex counts')
    plt.grid(linestyle='dotted', color='grey')
    fig.savefig(f'data/{DATA_DIR}/{OBJECT_NAME}_plot.png', bbox_inches="tight")
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')

args = parser.parse_args()

if args.Load:
    timings = load_results(DATA_DIR)
    plot_results(timings)
elif args.Save:
    timings = measure_time()
    save_results(timings, DATA_DIR)
    plot_results(timings)