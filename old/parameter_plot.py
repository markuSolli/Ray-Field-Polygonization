import utils
import ray_field

import trimesh
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from trimesh import Trimesh
from numpy import ndarray

DATA_DIR = 'parameter'
OBJECT_NAME = 'horse'
N_SAMPLES = 30000

def plot_results(points: list[int], depths: list[int], distances: list[list[float]]) -> None:
    fig, ax = plt.subplots()

    for i, depth in enumerate(distances):
        ax.plot(points, depth, label=f'{depths[i]}')
    
    ax.set_ylabel('Chamfer Distance')
    ax.set_ylim([0, 0.024])
    ax.set_xlim([0, 2000])
    ax.set_xlabel('N Points')
    #ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    ax.set_title(OBJECT_NAME)
    ax.legend(loc=(1.04, 0), title='Depth')
    plt.grid(linestyle='dotted', color='grey')
    fig.savefig(f'data/{DATA_DIR}/{OBJECT_NAME}_param_plot.png', bbox_inches="tight")
    plt.show()

def ray_field_polygonization() -> tuple[list[int], list[int], list[list[float]]]:
    original_mesh: Trimesh = ray_field.get_scaled_mesh(f'models/{OBJECT_NAME}.obj')

    n_points: list[int] = list(range(100, 2001, 100))
    depths: list[int] = list(range(5, 15))
    distances: list[list[float]] = []

    for d in depths:
        distances.append([])

    for n in n_points:
        print(f'Points:\t\t\t{n}')

        rays: ndarray = ray_field.generate_rays_between_sphere_points(n)

        intersect_locations, intersect_normals = ray_field.ray_intersection_with_mesh_batched(rays, original_mesh)

        for i in range(len(depths)):
            d: int = depths[i]

            generated_mesh = ray_field.poisson_surface_reconstruction(intersect_locations, intersect_normals, d)

            generated_points = np.asarray(generated_mesh.vertices).shape[0]

            original_samples = trimesh.sample.sample_surface_even(original_mesh, N_SAMPLES)[0]
            generated_samples = np.asarray(generated_mesh.sample_points_uniformly(N_SAMPLES).points)

            distance: float = utils.chamfer_distance(original_samples, generated_samples)

            print(f'{d}:\t{distance:.6f}\t{generated_points}')

            distances[i].append(distance)

        print("=====================")
    
    # Visualize generated mesh
    o3d.visualization.draw_geometries([generated_mesh])

    return n_points, depths, distances

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Save", action='store_true')
parser.add_argument("-l", "--Load", action='store_true')

args = parser.parse_args()

if args.Load:
    n_points, depths, distances = ray_field.load_results(DATA_DIR, OBJECT_NAME)
    plot_results(n_points, depths, distances)
elif args.Save:
    n_points, depths, distances = ray_field_polygonization()
    ray_field.save_results(n_points, depths, distances, DATA_DIR, OBJECT_NAME)
    plot_results(n_points, depths, distances)